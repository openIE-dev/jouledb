//! Conjugate Gradient iterative solvers for linear systems Ax = b.
//!
//! Standard CG for symmetric positive-definite systems, preconditioned CG
//! with Jacobi (diagonal) preconditioner, and BiCGSTAB for non-symmetric
//! systems.  Convergence history, residual tracking, tolerance control.

// ── Vector helpers ────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn axpy(a: f64, x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y.iter()).map(|(xi, yi)| a * xi + yi).collect()
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
}

fn vec_scale(s: f64, v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| s * x).collect()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
}

// ── Matrix-vector via closure ─────────────────────────────────

/// Result returned by iterative solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    /// The solution vector x.
    pub x: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver converged (residual below tolerance).
    pub converged: bool,
    /// Final relative residual norm.
    pub residual_norm: f64,
    /// Residual norm at each iteration.
    pub history: Vec<f64>,
}

// ── CG solver ─────────────────────────────────────────────────

/// Configuration for iterative solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on relative residual norm.
    pub tol: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
        }
    }
}

/// Conjugate Gradient solver for Ax = b where A is symmetric positive-definite.
///
/// `matvec` computes A * v.  If `x0` is `None`, the zero vector is used.
pub fn conjugate_gradient<F>(
    matvec: F,
    b: &[f64],
    x0: Option<&[f64]>,
    config: &SolverConfig,
) -> SolveResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let mut x = match x0 {
        Some(v) => v.to_vec(),
        None => vec![0.0; n],
    };

    let ax = matvec(&x);
    let mut r = vec_sub(b, &ax);
    let mut p = r.clone();
    let b_norm = norm2(b);
    if b_norm < 1e-30 {
        return SolveResult {
            x: vec![0.0; n],
            iterations: 0,
            converged: true,
            residual_norm: 0.0,
            history: vec![0.0],
        };
    }

    let mut rs_old = dot(&r, &r);
    let mut history = vec![rs_old.sqrt() / b_norm];

    for it in 0..config.max_iter {
        let ap = matvec(&p);
        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-30 {
            return SolveResult {
                x,
                iterations: it,
                converged: history.last().map_or(false, |h| *h < config.tol),
                residual_norm: *history.last().unwrap_or(&0.0),
                history,
            };
        }
        let alpha = rs_old / p_ap;
        x = axpy(alpha, &p, &x);
        r = axpy(-alpha, &ap, &r);
        let rs_new = dot(&r, &r);
        let rel = rs_new.sqrt() / b_norm;
        history.push(rel);
        if rel < config.tol {
            return SolveResult {
                x,
                iterations: it + 1,
                converged: true,
                residual_norm: rel,
                history,
            };
        }
        let beta = rs_new / rs_old;
        p = axpy(beta, &p, &r);
        rs_old = rs_new;
    }

    SolveResult {
        x,
        iterations: config.max_iter,
        converged: false,
        residual_norm: *history.last().unwrap_or(&0.0),
        history,
    }
}

/// Preconditioned Conjugate Gradient with Jacobi (diagonal) preconditioner.
///
/// `diag` contains the diagonal of A.  If a diagonal entry is near zero, 1.0 is used.
pub fn preconditioned_cg<F>(
    matvec: F,
    b: &[f64],
    diag: &[f64],
    x0: Option<&[f64]>,
    config: &SolverConfig,
) -> SolveResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let inv_diag: Vec<f64> = diag
        .iter()
        .map(|d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 })
        .collect();

    let apply_precond = |r: &[f64]| -> Vec<f64> {
        r.iter()
            .zip(inv_diag.iter())
            .map(|(ri, mi)| ri * mi)
            .collect()
    };

    let mut x = match x0 {
        Some(v) => v.to_vec(),
        None => vec![0.0; n],
    };

    let ax = matvec(&x);
    let mut r = vec_sub(b, &ax);
    let mut z = apply_precond(&r);
    let mut p = z.clone();
    let b_norm = norm2(b);
    if b_norm < 1e-30 {
        return SolveResult {
            x: vec![0.0; n],
            iterations: 0,
            converged: true,
            residual_norm: 0.0,
            history: vec![0.0],
        };
    }

    let mut rz_old = dot(&r, &z);
    let mut history = vec![norm2(&r) / b_norm];

    for it in 0..config.max_iter {
        let ap = matvec(&p);
        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-30 {
            return SolveResult {
                x,
                iterations: it,
                converged: history.last().map_or(false, |h| *h < config.tol),
                residual_norm: *history.last().unwrap_or(&0.0),
                history,
            };
        }
        let alpha = rz_old / p_ap;
        x = axpy(alpha, &p, &x);
        r = axpy(-alpha, &ap, &r);
        let rel = norm2(&r) / b_norm;
        history.push(rel);
        if rel < config.tol {
            return SolveResult {
                x,
                iterations: it + 1,
                converged: true,
                residual_norm: rel,
                history,
            };
        }
        z = apply_precond(&r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;
        p = axpy(beta, &p, &z);
        rz_old = rz_new;
    }

    SolveResult {
        x,
        iterations: config.max_iter,
        converged: false,
        residual_norm: *history.last().unwrap_or(&0.0),
        history,
    }
}

/// BiCGSTAB solver for non-symmetric systems Ax = b.
pub fn bicgstab<F>(
    matvec: F,
    b: &[f64],
    x0: Option<&[f64]>,
    config: &SolverConfig,
) -> SolveResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let mut x = match x0 {
        Some(v) => v.to_vec(),
        None => vec![0.0; n],
    };

    let b_norm = norm2(b);
    if b_norm < 1e-30 {
        return SolveResult {
            x: vec![0.0; n],
            iterations: 0,
            converged: true,
            residual_norm: 0.0,
            history: vec![0.0],
        };
    }

    let ax = matvec(&x);
    let mut r = vec_sub(b, &ax);
    let r0_hat = r.clone();
    let mut p = r.clone();
    let mut rho = dot(&r0_hat, &r);
    let mut history = vec![norm2(&r) / b_norm];

    for it in 0..config.max_iter {
        let v = matvec(&p);
        let r0v = dot(&r0_hat, &v);
        if r0v.abs() < 1e-30 {
            return SolveResult {
                x,
                iterations: it,
                converged: false,
                residual_norm: *history.last().unwrap_or(&0.0),
                history,
            };
        }
        let alpha = rho / r0v;
        let s = axpy(-alpha, &v, &r);
        let s_norm = norm2(&s);
        if s_norm / b_norm < config.tol {
            x = axpy(alpha, &p, &x);
            history.push(s_norm / b_norm);
            return SolveResult {
                x,
                iterations: it + 1,
                converged: true,
                residual_norm: s_norm / b_norm,
                history,
            };
        }

        let t = matvec(&s);
        let t_dot_t = dot(&t, &t);
        let omega = if t_dot_t.abs() > 1e-30 {
            dot(&t, &s) / t_dot_t
        } else {
            0.0
        };

        x = vec_add(&axpy(alpha, &p, &x), &vec_scale(omega, &s));
        r = axpy(-omega, &t, &s);

        let rel = norm2(&r) / b_norm;
        history.push(rel);
        if rel < config.tol {
            return SolveResult {
                x,
                iterations: it + 1,
                converged: true,
                residual_norm: rel,
                history,
            };
        }

        let rho_new = dot(&r0_hat, &r);
        if rho.abs() < 1e-30 || omega.abs() < 1e-30 {
            return SolveResult {
                x,
                iterations: it + 1,
                converged: false,
                residual_norm: rel,
                history,
            };
        }
        let beta = (rho_new / rho) * (alpha / omega);
        p = vec_add(&r, &vec_scale(beta, &vec_sub(&p, &vec_scale(omega, &v))));
        rho = rho_new;
    }

    SolveResult {
        x,
        iterations: config.max_iter,
        converged: false,
        residual_norm: *history.last().unwrap_or(&0.0),
        history,
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn approx_vec(a: &[f64], b: &[f64], eps: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    /// Simple SPD matrix: [[4,1],[1,3]]
    fn spd_matvec(v: &[f64]) -> Vec<f64> {
        vec![4.0 * v[0] + 1.0 * v[1], 1.0 * v[0] + 3.0 * v[1]]
    }

    /// 3x3 SPD: [[6,2,1],[2,5,2],[1,2,4]]
    fn spd3_matvec(v: &[f64]) -> Vec<f64> {
        vec![
            6.0 * v[0] + 2.0 * v[1] + 1.0 * v[2],
            2.0 * v[0] + 5.0 * v[1] + 2.0 * v[2],
            1.0 * v[0] + 2.0 * v[1] + 4.0 * v[2],
        ]
    }

    /// Non-symmetric: [[2,1],[0,3]]
    fn nonsym_matvec(v: &[f64]) -> Vec<f64> {
        vec![2.0 * v[0] + 1.0 * v[1], 3.0 * v[1]]
    }

    #[test]
    fn test_cg_simple() {
        let b = vec![1.0, 2.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = conjugate_gradient(spd_matvec, &b, None, &cfg);
        assert!(res.converged);
        // Verify Ax = b
        let ax = spd_matvec(&res.x);
        assert!(approx_vec(&ax, &b, 1e-6));
    }

    #[test]
    fn test_cg_known_solution() {
        // [[4,1],[1,3]] x = [1,1] => b = [5,4]
        let b = vec![5.0, 4.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = conjugate_gradient(spd_matvec, &b, None, &cfg);
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[1.0, 1.0], 1e-6));
    }

    #[test]
    fn test_cg_3x3() {
        let b = vec![1.0, 2.0, 3.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = conjugate_gradient(spd3_matvec, &b, None, &cfg);
        assert!(res.converged);
        let ax = spd3_matvec(&res.x);
        assert!(approx_vec(&ax, &b, 1e-6));
    }

    #[test]
    fn test_cg_with_initial_guess() {
        let b = vec![5.0, 4.0];
        let x0 = vec![0.5, 0.5];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = conjugate_gradient(spd_matvec, &b, Some(&x0), &cfg);
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[1.0, 1.0], 1e-6));
    }

    #[test]
    fn test_cg_zero_rhs() {
        let b = vec![0.0, 0.0];
        let cfg = SolverConfig::default();
        let res = conjugate_gradient(spd_matvec, &b, None, &cfg);
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[0.0, 0.0], 1e-10));
    }

    #[test]
    fn test_cg_convergence_history() {
        let b = vec![1.0, 2.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = conjugate_gradient(spd_matvec, &b, None, &cfg);
        assert!(res.converged);
        assert!(!res.history.is_empty());
        // Residuals should be monotonically (roughly) decreasing
        for w in res.history.windows(2) {
            // Allow very small increases due to floating point
            assert!(w[1] <= w[0] + 1e-10);
        }
    }

    #[test]
    fn test_cg_identity_matvec() {
        let id_matvec = |v: &[f64]| v.to_vec();
        let b = vec![3.0, 7.0, 11.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = conjugate_gradient(id_matvec, &b, None, &cfg);
        assert!(res.converged);
        assert!(approx_vec(&res.x, &b, 1e-6));
    }

    #[test]
    fn test_pcg_simple() {
        let b = vec![1.0, 2.0];
        let diag = vec![4.0, 3.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = preconditioned_cg(spd_matvec, &b, &diag, None, &cfg);
        assert!(res.converged);
        let ax = spd_matvec(&res.x);
        assert!(approx_vec(&ax, &b, 1e-6));
    }

    #[test]
    fn test_pcg_3x3() {
        let b = vec![1.0, 2.0, 3.0];
        let diag = vec![6.0, 5.0, 4.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = preconditioned_cg(spd3_matvec, &b, &diag, None, &cfg);
        assert!(res.converged);
        let ax = spd3_matvec(&res.x);
        assert!(approx_vec(&ax, &b, 1e-6));
    }

    #[test]
    fn test_pcg_zero_rhs() {
        let b = vec![0.0, 0.0];
        let diag = vec![4.0, 3.0];
        let cfg = SolverConfig::default();
        let res = preconditioned_cg(spd_matvec, &b, &diag, None, &cfg);
        assert!(res.converged);
    }

    #[test]
    fn test_pcg_with_initial_guess() {
        let b = vec![5.0, 4.0];
        let diag = vec![4.0, 3.0];
        let x0 = vec![0.9, 0.9];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = preconditioned_cg(spd_matvec, &b, &diag, Some(&x0), &cfg);
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[1.0, 1.0], 1e-6));
    }

    #[test]
    fn test_bicgstab_nonsymmetric() {
        // [[2,1],[0,3]] x = [1, 2/3] => b = [2+2/3, 2] = [8/3, 2]
        let b = vec![8.0 / 3.0, 2.0];
        let cfg = SolverConfig { max_iter: 200, tol: 1e-10 };
        let res = bicgstab(nonsym_matvec, &b, None, &cfg);
        assert!(res.converged);
        let ax = nonsym_matvec(&res.x);
        assert!(approx_vec(&ax, &b, 1e-6));
    }

    #[test]
    fn test_bicgstab_symmetric_system() {
        // BiCGSTAB should also work on SPD systems.
        let b = vec![5.0, 4.0];
        let cfg = SolverConfig { max_iter: 200, tol: 1e-10 };
        let res = bicgstab(spd_matvec, &b, None, &cfg);
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[1.0, 1.0], 1e-6));
    }

    #[test]
    fn test_bicgstab_zero_rhs() {
        let b = vec![0.0, 0.0];
        let cfg = SolverConfig::default();
        let res = bicgstab(nonsym_matvec, &b, None, &cfg);
        assert!(res.converged);
    }

    #[test]
    fn test_bicgstab_3x3() {
        // Non-symmetric 3x3: [[3,1,0],[0,4,1],[1,0,2]]
        let mv = |v: &[f64]| {
            vec![
                3.0 * v[0] + v[1],
                4.0 * v[1] + v[2],
                v[0] + 2.0 * v[2],
            ]
        };
        let b = vec![4.0, 5.0, 3.0];
        let cfg = SolverConfig { max_iter: 200, tol: 1e-10 };
        let res = bicgstab(mv, &b, None, &cfg);
        assert!(res.converged);
        let ax = mv(&res.x);
        assert!(approx_vec(&ax, &b, 1e-6));
    }

    #[test]
    fn test_bicgstab_with_initial_guess() {
        let b = vec![5.0, 4.0];
        let x0 = vec![1.1, 0.9];
        let cfg = SolverConfig { max_iter: 200, tol: 1e-10 };
        let res = bicgstab(spd_matvec, &b, Some(&x0), &cfg);
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[1.0, 1.0], 1e-6));
    }

    #[test]
    fn test_cg_max_iter_exceeded() {
        let b = vec![1.0, 2.0];
        let cfg = SolverConfig { max_iter: 1, tol: 1e-15 };
        let res = conjugate_gradient(spd_matvec, &b, None, &cfg);
        // With only 1 iteration, 2x2 SPD might or might not converge
        // but iterations should be limited
        assert!(res.iterations <= 1);
    }

    #[test]
    fn test_solver_result_fields() {
        let b = vec![1.0, 2.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let res = conjugate_gradient(spd_matvec, &b, None, &cfg);
        assert_eq!(res.x.len(), 2);
        assert!(res.residual_norm >= 0.0);
        assert!(res.history.len() >= 1);
    }

    #[test]
    fn test_default_config() {
        let cfg = SolverConfig::default();
        assert_eq!(cfg.max_iter, 1000);
        assert!(approx_eq(cfg.tol, 1e-10, 1e-15));
    }

    #[test]
    fn test_diagonal_system() {
        // Diagonal 3x3: trivial but verifies correctness
        let diag_mv = |v: &[f64]| vec![2.0 * v[0], 5.0 * v[1], 10.0 * v[2]];
        let b = vec![4.0, 15.0, 30.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-12 };
        let res = conjugate_gradient(diag_mv, &b, None, &cfg);
        assert!(res.converged);
        assert!(approx_vec(&res.x, &[2.0, 3.0, 3.0], 1e-6));
    }

    #[test]
    fn test_pcg_vs_cg_same_answer() {
        let b = vec![5.0, 4.0];
        let diag = vec![4.0, 3.0];
        let cfg = SolverConfig { max_iter: 100, tol: 1e-10 };
        let cg = conjugate_gradient(spd_matvec, &b, None, &cfg);
        let pcg = preconditioned_cg(spd_matvec, &b, &diag, None, &cfg);
        assert!(approx_vec(&cg.x, &pcg.x, 1e-6));
    }
}
