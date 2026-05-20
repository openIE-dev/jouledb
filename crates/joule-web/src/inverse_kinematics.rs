//! Inverse kinematics — Jacobian transpose, Jacobian pseudoinverse,
//! damped least squares (Levenberg-Marquardt), Cyclic Coordinate Descent (CCD),
//! and FABRIK (Forward And Backward Reaching Inverse Kinematics).
//!
//! Solves for joint values that place the end-effector at a desired target
//! pose or position.  All solvers are iterative and return convergence status.

use std::f64::consts::PI;

// ── Errors ──────────────────────────────────────────────────────

/// Errors produced by inverse-kinematics solvers.
#[derive(Debug, Clone, PartialEq)]
pub enum IkError {
    /// Target is unreachable (outside workspace).
    Unreachable { distance: f64, max_reach: f64 },
    /// Solver did not converge within the iteration limit.
    NotConverged { iterations: usize, residual: f64 },
    /// Singular Jacobian encountered.
    SingularJacobian,
    /// Invalid configuration (e.g., zero-length link).
    InvalidConfig(String),
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for IkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable { distance, max_reach } => {
                write!(f, "target unreachable: distance={distance:.4}, max_reach={max_reach:.4}")
            }
            Self::NotConverged { iterations, residual } => {
                write!(f, "IK not converged after {iterations} iters (residual={residual:.6})")
            }
            Self::SingularJacobian => write!(f, "singular Jacobian"),
            Self::InvalidConfig(msg) => write!(f, "invalid IK config: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for IkError {}

// ── IK Result ──────────────────────────────────────────────────

/// Result of an IK solve.
#[derive(Debug, Clone, PartialEq)]
pub struct IkSolution {
    /// Joint values that achieve (or approximate) the target.
    pub joints: Vec<f64>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual (position error norm).
    pub residual: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

impl std::fmt::Display for IkSolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IkSolution({} joints, {} iters, residual={:.6}, converged={})",
            self.joints.len(),
            self.iterations,
            self.residual,
            self.converged,
        )
    }
}

// ── Solver Configuration ───────────────────────────────────────

/// Solver method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IkMethod {
    /// Jacobian transpose.
    JacobianTranspose,
    /// Jacobian pseudoinverse.
    JacobianPseudoinverse,
    /// Damped least squares (Levenberg-Marquardt).
    DampedLeastSquares,
}

/// Configuration for iterative IK solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct IkConfig {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (position error).
    pub tolerance: f64,
    /// Step size / gain for Jacobian methods.
    pub step_size: f64,
    /// Damping factor for DLS.
    pub damping: f64,
    /// Finite-difference delta for numerical Jacobian.
    pub fd_delta: f64,
    /// Joint limits: `(min, max)` per joint.  If empty, no limits enforced.
    pub joint_limits: Vec<(f64, f64)>,
}

impl IkConfig {
    /// Default configuration.
    pub fn new() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-4,
            step_size: 0.1,
            damping: 0.01,
            fd_delta: 1e-6,
            joint_limits: Vec::new(),
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_step_size(mut self, s: f64) -> Self {
        self.step_size = s;
        self
    }

    pub fn with_damping(mut self, d: f64) -> Self {
        self.damping = d;
        self
    }

    pub fn with_joint_limits(mut self, limits: Vec<(f64, f64)>) -> Self {
        self.joint_limits = limits;
        self
    }
}

impl Default for IkConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Jacobian-based IK Solver ───────────────────────────────────

/// Generic Jacobian-based IK solver.
///
/// Requires a user-supplied forward-kinematics function that maps joint values
/// to a Cartesian position `[x, y, z]`.
#[derive(Debug, Clone)]
pub struct JacobianIkSolver {
    method: IkMethod,
    config: IkConfig,
}

impl JacobianIkSolver {
    /// Create a solver with the given method.
    pub fn new(method: IkMethod) -> Self {
        Self { method, config: IkConfig::new() }
    }

    /// Set configuration.
    pub fn with_config(mut self, config: IkConfig) -> Self {
        self.config = config;
        self
    }

    /// Solve for target position using the provided FK function.
    ///
    /// `fk` maps `&[f64]` joint values to `[f64; 3]` position.
    /// `q0` is the initial guess.
    pub fn solve<F>(&self, fk: F, target: [f64; 3], q0: &[f64]) -> Result<IkSolution, IkError>
    where
        F: Fn(&[f64]) -> [f64; 3],
    {
        let n = q0.len();
        if n == 0 {
            return Err(IkError::InvalidConfig("zero joints".into()));
        }
        let mut q = q0.to_vec();
        let mut residual = f64::MAX;

        for iter in 0..self.config.max_iterations {
            let pos = fk(&q);
            let err = [target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]];
            residual = (err[0] * err[0] + err[1] * err[1] + err[2] * err[2]).sqrt();

            if residual < self.config.tolerance {
                return Ok(IkSolution {
                    joints: q,
                    iterations: iter,
                    residual,
                    converged: true,
                });
            }

            // Numerical Jacobian (3 x n)
            let jac = self.numerical_jacobian_3xn(&fk, &q, n);

            // Compute delta_q based on method
            let dq = match self.method {
                IkMethod::JacobianTranspose => {
                    self.jacobian_transpose_step(&jac, &err, n)
                }
                IkMethod::JacobianPseudoinverse => {
                    self.jacobian_pseudoinverse_step(&jac, &err, n)
                }
                IkMethod::DampedLeastSquares => {
                    self.dls_step(&jac, &err, n)
                }
            };

            for i in 0..n {
                q[i] += self.config.step_size * dq[i];
            }
            self.clamp_joints(&mut q);
        }

        Err(IkError::NotConverged {
            iterations: self.config.max_iterations,
            residual,
        })
    }

    /// Compute 3 x n numerical Jacobian.
    fn numerical_jacobian_3xn<F>(&self, fk: &F, q: &[f64], n: usize) -> Vec<f64>
    where
        F: Fn(&[f64]) -> [f64; 3],
    {
        let mut jac = vec![0.0; 3 * n];
        let p0 = fk(q);
        let mut q_pert = q.to_vec();
        for j in 0..n {
            let orig = q_pert[j];
            q_pert[j] = orig + self.config.fd_delta;
            let p1 = fk(&q_pert);
            for r in 0..3 {
                jac[r * n + j] = (p1[r] - p0[r]) / self.config.fd_delta;
            }
            q_pert[j] = orig;
        }
        jac
    }

    /// J^T * e (scaled).
    fn jacobian_transpose_step(&self, jac: &[f64], err: &[f64; 3], n: usize) -> Vec<f64> {
        let mut dq = vec![0.0; n];
        for j in 0..n {
            for r in 0..3 {
                dq[j] += jac[r * n + j] * err[r];
            }
        }
        dq
    }

    /// Pseudoinverse step: J^T (J J^T)^{-1} e.
    fn jacobian_pseudoinverse_step(
        &self,
        jac: &[f64],
        err: &[f64; 3],
        n: usize,
    ) -> Vec<f64> {
        // J J^T is 3x3
        let jjt = self.mul_jjt(jac, n);
        let jjt_inv = match invert_3x3(&jjt) {
            Some(inv) => inv,
            None => return self.jacobian_transpose_step(jac, err, n), // fallback
        };
        // J^T * (JJT_inv * e)
        let mut tmp = [0.0; 3];
        for r in 0..3 {
            for c in 0..3 {
                tmp[r] += jjt_inv[r * 3 + c] * err[c];
            }
        }
        let mut dq = vec![0.0; n];
        for j in 0..n {
            for r in 0..3 {
                dq[j] += jac[r * n + j] * tmp[r];
            }
        }
        dq
    }

    /// Damped least squares: J^T (J J^T + lambda^2 I)^{-1} e.
    fn dls_step(&self, jac: &[f64], err: &[f64; 3], n: usize) -> Vec<f64> {
        let mut jjt = self.mul_jjt(jac, n);
        let lam2 = self.config.damping * self.config.damping;
        jjt[0] += lam2;
        jjt[4] += lam2;
        jjt[8] += lam2;

        let jjt_inv = match invert_3x3(&jjt) {
            Some(inv) => inv,
            None => return self.jacobian_transpose_step(jac, err, n),
        };
        let mut tmp = [0.0; 3];
        for r in 0..3 {
            for c in 0..3 {
                tmp[r] += jjt_inv[r * 3 + c] * err[c];
            }
        }
        let mut dq = vec![0.0; n];
        for j in 0..n {
            for r in 0..3 {
                dq[j] += jac[r * n + j] * tmp[r];
            }
        }
        dq
    }

    /// Compute J * J^T (3x3).
    fn mul_jjt(&self, jac: &[f64], n: usize) -> [f64; 9] {
        let mut m = [0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                let mut s = 0.0;
                for k in 0..n {
                    s += jac[r * n + k] * jac[c * n + k];
                }
                m[r * 3 + c] = s;
            }
        }
        m
    }

    /// Clamp joints to configured limits.
    fn clamp_joints(&self, q: &mut [f64]) {
        if self.config.joint_limits.is_empty() {
            return;
        }
        for (i, val) in q.iter_mut().enumerate() {
            if let Some(&(lo, hi)) = self.config.joint_limits.get(i) {
                *val = val.clamp(lo, hi);
            }
        }
    }
}

impl std::fmt::Display for JacobianIkSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self.method {
            IkMethod::JacobianTranspose => "Jacobian Transpose",
            IkMethod::JacobianPseudoinverse => "Jacobian Pseudoinverse",
            IkMethod::DampedLeastSquares => "Damped Least Squares",
        };
        write!(
            f,
            "JacobianIK({name}, max_iter={}, tol={:.1e})",
            self.config.max_iterations, self.config.tolerance,
        )
    }
}

// ── CCD Solver ─────────────────────────────────────────────────

/// Cyclic Coordinate Descent (CCD) solver for 2D/3D planar chains.
///
/// Iterates from the last joint to the first, rotating each joint to
/// minimize the distance from end-effector to target.
#[derive(Debug, Clone)]
pub struct CcdSolver {
    /// Link lengths.
    link_lengths: Vec<f64>,
    config: IkConfig,
}

impl CcdSolver {
    /// Create from link lengths.
    pub fn new(link_lengths: Vec<f64>) -> Result<Self, IkError> {
        if link_lengths.is_empty() {
            return Err(IkError::InvalidConfig("no links".into()));
        }
        for (i, &l) in link_lengths.iter().enumerate() {
            if l <= 0.0 {
                return Err(IkError::InvalidConfig(format!("link {i} has non-positive length {l}")));
            }
        }
        Ok(Self { link_lengths, config: IkConfig::new() })
    }

    pub fn with_config(mut self, config: IkConfig) -> Self {
        self.config = config;
        self
    }

    /// Total reach of the chain.
    pub fn max_reach(&self) -> f64 {
        self.link_lengths.iter().sum()
    }

    /// Solve for a 2D target position.
    pub fn solve_2d(&self, target: [f64; 2], q0: &[f64]) -> Result<IkSolution, IkError> {
        let n = self.link_lengths.len();
        if q0.len() != n {
            return Err(IkError::DimensionMismatch { expected: n, got: q0.len() });
        }
        let dist = (target[0] * target[0] + target[1] * target[1]).sqrt();
        let reach = self.max_reach();
        if dist > reach {
            return Err(IkError::Unreachable { distance: dist, max_reach: reach });
        }

        let mut angles = q0.to_vec();
        let mut residual = f64::MAX;

        for iter in 0..self.config.max_iterations {
            // Forward pass: compute joint positions
            let positions = self.compute_positions_2d(&angles);
            let ee = positions[n];
            let err_x = target[0] - ee[0];
            let err_y = target[1] - ee[1];
            residual = (err_x * err_x + err_y * err_y).sqrt();

            if residual < self.config.tolerance {
                return Ok(IkSolution {
                    joints: angles,
                    iterations: iter,
                    residual,
                    converged: true,
                });
            }

            // CCD: iterate from tip to base
            for j in (0..n).rev() {
                let positions = self.compute_positions_2d(&angles);
                let joint_pos = positions[j];
                let ee_pos = positions[n];

                let to_ee = (ee_pos[1] - joint_pos[1]).atan2(ee_pos[0] - joint_pos[0]);
                let to_target = (target[1] - joint_pos[1]).atan2(target[0] - joint_pos[0]);

                let mut delta = to_target - to_ee;
                // Wrap to [-PI, PI]
                while delta > PI { delta -= 2.0 * PI; }
                while delta < -PI { delta += 2.0 * PI; }

                angles[j] += delta;
                self.clamp_angle(&mut angles[j], j);
            }
        }

        Err(IkError::NotConverged {
            iterations: self.config.max_iterations,
            residual,
        })
    }

    /// Compute 2D joint positions from angles.
    fn compute_positions_2d(&self, angles: &[f64]) -> Vec<[f64; 2]> {
        let n = self.link_lengths.len();
        let mut positions = Vec::with_capacity(n + 1);
        positions.push([0.0, 0.0]);
        let mut cumulative_angle = 0.0;
        for i in 0..n {
            cumulative_angle += angles[i];
            let prev = positions[i];
            let x = prev[0] + self.link_lengths[i] * cumulative_angle.cos();
            let y = prev[1] + self.link_lengths[i] * cumulative_angle.sin();
            positions.push([x, y]);
        }
        positions
    }

    fn clamp_angle(&self, angle: &mut f64, idx: usize) {
        if let Some(&(lo, hi)) = self.config.joint_limits.get(idx) {
            *angle = angle.clamp(lo, hi);
        }
    }
}

impl std::fmt::Display for CcdSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CCD({} links, reach={:.3})", self.link_lengths.len(), self.max_reach())
    }
}

// ── FABRIK Solver ──────────────────────────────────────────────

/// Forward And Backward Reaching Inverse Kinematics (FABRIK) solver.
///
/// Operates on a chain of 2D joint positions with fixed link lengths.
/// Alternates forward and backward reaching passes until convergence.
#[derive(Debug, Clone)]
pub struct FabrikSolver {
    link_lengths: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
}

impl FabrikSolver {
    /// Create from link lengths.
    pub fn new(link_lengths: Vec<f64>) -> Result<Self, IkError> {
        if link_lengths.is_empty() {
            return Err(IkError::InvalidConfig("no links".into()));
        }
        Ok(Self {
            link_lengths,
            tolerance: 1e-4,
            max_iterations: 100,
        })
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Total chain reach.
    pub fn max_reach(&self) -> f64 {
        self.link_lengths.iter().sum()
    }

    /// Solve for a 2D target.  Returns the joint positions (n+1 points).
    pub fn solve_2d(
        &self,
        target: [f64; 2],
        initial_positions: &[[f64; 2]],
    ) -> Result<FabrikResult, IkError> {
        let n = self.link_lengths.len();
        if initial_positions.len() != n + 1 {
            return Err(IkError::DimensionMismatch {
                expected: n + 1,
                got: initial_positions.len(),
            });
        }
        let dist = (target[0] * target[0] + target[1] * target[1]).sqrt();
        let reach = self.max_reach();
        if dist > reach + self.tolerance {
            // Stretch toward target
            let mut positions = vec![[0.0; 2]; n + 1];
            positions[0] = initial_positions[0];
            let dir = normalize_2d(target[0], target[1]);
            let mut cum = 0.0;
            for i in 0..n {
                cum += self.link_lengths[i];
                positions[i + 1] = [dir[0] * cum, dir[1] * cum];
            }
            let ee = positions[n];
            let residual = dist_2d(ee, target);
            return Ok(FabrikResult {
                positions,
                iterations: 0,
                residual,
                converged: residual < self.tolerance,
            });
        }

        let mut positions: Vec<[f64; 2]> = initial_positions.to_vec();
        let base = positions[0];

        for iter in 0..self.max_iterations {
            let ee = positions[n];
            let residual = dist_2d(ee, target);
            if residual < self.tolerance {
                return Ok(FabrikResult {
                    positions,
                    iterations: iter,
                    residual,
                    converged: true,
                });
            }

            // Forward reaching: set end to target, work backward
            positions[n] = target;
            for i in (0..n).rev() {
                let d = dist_2d(positions[i], positions[i + 1]);
                let ratio = if d > 1e-12 { self.link_lengths[i] / d } else { 1.0 };
                positions[i] = [
                    positions[i + 1][0] + ratio * (positions[i][0] - positions[i + 1][0]),
                    positions[i + 1][1] + ratio * (positions[i][1] - positions[i + 1][1]),
                ];
            }

            // Backward reaching: set base to original, work forward
            positions[0] = base;
            for i in 0..n {
                let d = dist_2d(positions[i], positions[i + 1]);
                let ratio = if d > 1e-12 { self.link_lengths[i] / d } else { 1.0 };
                positions[i + 1] = [
                    positions[i][0] + ratio * (positions[i + 1][0] - positions[i][0]),
                    positions[i][1] + ratio * (positions[i + 1][1] - positions[i][1]),
                ];
            }
        }

        let ee = positions[n];
        let residual = dist_2d(ee, target);
        Err(IkError::NotConverged {
            iterations: self.max_iterations,
            residual,
        })
    }
}

impl std::fmt::Display for FabrikSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FABRIK({} links, reach={:.3}, tol={:.1e})",
            self.link_lengths.len(),
            self.max_reach(),
            self.tolerance,
        )
    }
}

/// Result from a FABRIK solve.
#[derive(Debug, Clone, PartialEq)]
pub struct FabrikResult {
    /// Joint positions (n+1 points for n links).
    pub positions: Vec<[f64; 2]>,
    /// Iterations used.
    pub iterations: usize,
    /// Final residual.
    pub residual: f64,
    /// Whether solver converged.
    pub converged: bool,
}

impl std::fmt::Display for FabrikResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FabrikResult({} points, {} iters, residual={:.6}, converged={})",
            self.positions.len(),
            self.iterations,
            self.residual,
            self.converged,
        )
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Invert a 3x3 matrix (row-major).  Returns `None` if singular.
fn invert_3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-15 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}

fn dist_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn normalize_2d(x: f64, y: f64) -> [f64; 2] {
    let len = (x * x + y * y).sqrt();
    if len < 1e-15 {
        [1.0, 0.0]
    } else {
        [x / len, y / len]
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-3;

    fn fk_2r(q: &[f64]) -> [f64; 3] {
        let l1 = 1.0;
        let l2 = 1.0;
        let x = l1 * q[0].cos() + l2 * (q[0] + q[1]).cos();
        let y = l1 * q[0].sin() + l2 * (q[0] + q[1]).sin();
        [x, y, 0.0]
    }

    #[test]
    fn test_jacobian_transpose_2r() {
        let solver = JacobianIkSolver::new(IkMethod::JacobianTranspose)
            .with_config(IkConfig::new().with_max_iterations(500).with_step_size(0.5));
        let result = solver.solve(fk_2r, [1.0, 1.0, 0.0], &[0.0, 0.0]).unwrap();
        assert!(result.converged);
        let pos = fk_2r(&result.joints);
        assert!((pos[0] - 1.0).abs() < EPS);
        assert!((pos[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_pseudoinverse_2r() {
        let solver = JacobianIkSolver::new(IkMethod::JacobianPseudoinverse)
            .with_config(IkConfig::new().with_max_iterations(300).with_step_size(0.5));
        let result = solver.solve(fk_2r, [1.5, 0.5, 0.0], &[0.1, 0.1]).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_dls_2r() {
        let solver = JacobianIkSolver::new(IkMethod::DampedLeastSquares)
            .with_config(IkConfig::new().with_damping(0.05).with_max_iterations(300));
        let result = solver.solve(fk_2r, [0.5, 1.5, 0.0], &[0.0, 0.0]).unwrap();
        assert!(result.converged);
        let pos = fk_2r(&result.joints);
        assert!((pos[0] - 0.5).abs() < EPS);
        assert!((pos[1] - 1.5).abs() < EPS);
    }

    #[test]
    fn test_ccd_2d_reachable() {
        let solver = CcdSolver::new(vec![1.0, 1.0, 1.0]).unwrap()
            .with_config(IkConfig::new().with_max_iterations(200));
        let result = solver.solve_2d([2.0, 1.0], &[0.0, 0.0, 0.0]).unwrap();
        assert!(result.converged);
        let positions = solver.compute_positions_2d(&result.joints);
        let ee = positions[3];
        assert!((ee[0] - 2.0).abs() < EPS);
        assert!((ee[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_ccd_unreachable() {
        let solver = CcdSolver::new(vec![1.0, 1.0]).unwrap();
        let result = solver.solve_2d([5.0, 0.0], &[0.0, 0.0]);
        assert!(matches!(result, Err(IkError::Unreachable { .. })));
    }

    #[test]
    fn test_ccd_dimension_mismatch() {
        let solver = CcdSolver::new(vec![1.0, 1.0]).unwrap();
        let result = solver.solve_2d([1.0, 0.0], &[0.0]);
        assert!(matches!(result, Err(IkError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_fabrik_reachable() {
        let solver = FabrikSolver::new(vec![1.0, 1.0]).unwrap();
        let init = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let result = solver.solve_2d([1.0, 1.0], &init).unwrap();
        assert!(result.converged);
        let ee = result.positions[2];
        assert!((ee[0] - 1.0).abs() < EPS);
        assert!((ee[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_fabrik_at_origin() {
        let solver = FabrikSolver::new(vec![1.0, 1.0]).unwrap()
            .with_tolerance(1e-3);
        let init = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let result = solver.solve_2d([0.0, 0.0], &init).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_fabrik_unreachable_stretches() {
        let solver = FabrikSolver::new(vec![1.0, 1.0]).unwrap();
        let init = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let result = solver.solve_2d([10.0, 0.0], &init).unwrap();
        // Should stretch toward target but not converge
        assert!(result.positions[2][0] > 1.5);
    }

    #[test]
    fn test_fabrik_dimension_error() {
        let solver = FabrikSolver::new(vec![1.0]).unwrap();
        let result = solver.solve_2d([0.5, 0.0], &[[0.0, 0.0]]);
        assert!(matches!(result, Err(IkError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ik_solution_display() {
        let sol = IkSolution {
            joints: vec![0.5, -0.3],
            iterations: 42,
            residual: 0.001,
            converged: true,
        };
        let s = format!("{sol}");
        assert!(s.contains("2 joints"));
        assert!(s.contains("42 iters"));
    }

    #[test]
    fn test_ik_error_display() {
        let e = IkError::Unreachable { distance: 5.0, max_reach: 3.0 };
        let s = format!("{e}");
        assert!(s.contains("unreachable"));
    }

    #[test]
    fn test_jacobian_solver_display() {
        let solver = JacobianIkSolver::new(IkMethod::DampedLeastSquares);
        let s = format!("{solver}");
        assert!(s.contains("Damped Least Squares"));
    }

    #[test]
    fn test_ccd_display() {
        let solver = CcdSolver::new(vec![1.0, 2.0]).unwrap();
        let s = format!("{solver}");
        assert!(s.contains("CCD"));
        assert!(s.contains("3.000"));
    }

    #[test]
    fn test_fabrik_display() {
        let solver = FabrikSolver::new(vec![1.0, 1.5]).unwrap();
        let s = format!("{solver}");
        assert!(s.contains("FABRIK"));
    }

    #[test]
    fn test_invert_3x3_identity() {
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert_3x3(&id).unwrap();
        for i in 0..9 {
            assert!((inv[i] - id[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_invert_3x3_singular() {
        let m = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(invert_3x3(&m).is_none());
    }

    #[test]
    fn test_ik_config_defaults() {
        let cfg = IkConfig::new();
        assert_eq!(cfg.max_iterations, 200);
        assert!((cfg.tolerance - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_zero_joints_error() {
        let solver = JacobianIkSolver::new(IkMethod::JacobianTranspose);
        let result = solver.solve(|_: &[f64]| [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &[]);
        assert!(matches!(result, Err(IkError::InvalidConfig(_))));
    }
}
