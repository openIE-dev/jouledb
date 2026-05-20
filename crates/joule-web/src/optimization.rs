//! Numerical Optimization — gradient descent, SGD, Adam, line search,
//! Newton's method, golden section search, convergence criteria, and
//! learning rate schedules.
//!
//! Pure Rust — no external numeric or optimization dependencies.

use std::fmt;

// ── Convergence Criteria ────────────────────────────────────────

/// Convergence check result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Not yet converged.
    Running,
    /// Converged by function value tolerance.
    ConvergedValue,
    /// Converged by gradient norm tolerance.
    ConvergedGradient,
    /// Converged by parameter change tolerance.
    ConvergedParam,
    /// Hit maximum iterations.
    MaxIterations,
}

/// Convergence criteria for optimization.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Tolerance on function value change.
    pub ftol: f64,
    /// Tolerance on gradient norm.
    pub gtol: f64,
    /// Tolerance on parameter change.
    pub xtol: f64,
    /// Maximum iterations.
    pub max_iter: usize,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            ftol: 1e-8,
            gtol: 1e-6,
            xtol: 1e-8,
            max_iter: 1000,
        }
    }
}

impl ConvergenceCriteria {
    /// Check convergence given current state.
    pub fn check(
        &self,
        iter: usize,
        f_prev: f64,
        f_curr: f64,
        grad_norm: f64,
        param_change: f64,
    ) -> ConvergenceStatus {
        if iter >= self.max_iter {
            return ConvergenceStatus::MaxIterations;
        }
        if (f_prev - f_curr).abs() < self.ftol {
            return ConvergenceStatus::ConvergedValue;
        }
        if grad_norm < self.gtol {
            return ConvergenceStatus::ConvergedGradient;
        }
        if param_change < self.xtol {
            return ConvergenceStatus::ConvergedParam;
        }
        ConvergenceStatus::Running
    }
}

// ── Learning Rate Schedules ─────────────────────────────────────

/// Learning rate schedule types.
#[derive(Debug, Clone)]
pub enum LrSchedule {
    /// Constant learning rate.
    Constant(f64),
    /// Step decay: lr * decay^(step / step_size).
    Step { initial_lr: f64, decay: f64, step_size: usize },
    /// Exponential decay: lr * exp(-decay * step).
    Exponential { initial_lr: f64, decay: f64 },
    /// Cosine annealing: lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * step / total)).
    Cosine { lr_max: f64, lr_min: f64, total_steps: usize },
}

impl LrSchedule {
    /// Get the learning rate at a given step.
    pub fn get_lr(&self, step: usize) -> f64 {
        match self {
            Self::Constant(lr) => *lr,
            Self::Step { initial_lr, decay, step_size } => {
                initial_lr * decay.powi((step / step_size) as i32)
            }
            Self::Exponential { initial_lr, decay } => {
                initial_lr * (-decay * step as f64).exp()
            }
            Self::Cosine { lr_max, lr_min, total_steps } => {
                let t = if *total_steps == 0 { 0.0 } else { step as f64 / *total_steps as f64 };
                lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
            }
        }
    }
}

impl fmt::Display for LrSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(lr) => write!(f, "Constant(lr={})", lr),
            Self::Step { initial_lr, decay, step_size } =>
                write!(f, "Step(lr={}, decay={}, step={})", initial_lr, decay, step_size),
            Self::Exponential { initial_lr, decay } =>
                write!(f, "Exponential(lr={}, decay={})", initial_lr, decay),
            Self::Cosine { lr_max, lr_min, total_steps } =>
                write!(f, "Cosine(max={}, min={}, steps={})", lr_max, lr_min, total_steps),
        }
    }
}

// ── Optimization Result ─────────────────────────────────────────

/// Result of an optimization run.
#[derive(Debug, Clone)]
pub struct OptResult {
    /// Final parameter values.
    pub params: Vec<f64>,
    /// Final function value.
    pub f_val: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Convergence status.
    pub status: ConvergenceStatus,
    /// History of function values.
    pub f_history: Vec<f64>,
}

// ── Numerical Gradient ──────────────────────────────────────────

/// Compute numerical gradient using central differences.
pub fn numerical_gradient(f: &dyn Fn(&[f64]) -> f64, params: &[f64], eps: f64) -> Vec<f64> {
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut params_plus = params.to_vec();
    let mut params_minus = params.to_vec();

    for i in 0..n {
        params_plus[i] = params[i] + eps;
        params_minus[i] = params[i] - eps;
        grad[i] = (f(&params_plus) - f(&params_minus)) / (2.0 * eps);
        params_plus[i] = params[i];
        params_minus[i] = params[i];
    }
    grad
}

// ── Gradient Descent ────────────────────────────────────────────

/// Standard gradient descent optimization.
pub fn gradient_descent(
    f: &dyn Fn(&[f64]) -> f64,
    grad_f: &dyn Fn(&[f64]) -> Vec<f64>,
    initial_params: &[f64],
    schedule: &LrSchedule,
    criteria: &ConvergenceCriteria,
) -> OptResult {
    let mut params = initial_params.to_vec();
    let mut f_val = f(&params);
    let mut f_history = vec![f_val];

    for iter in 0..criteria.max_iter {
        let grad = grad_f(&params);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let lr = schedule.get_lr(iter);

        let mut new_params = params.clone();
        let mut param_change = 0.0;
        for (i, g) in grad.iter().enumerate() {
            let delta = lr * g;
            new_params[i] -= delta;
            param_change += delta * delta;
        }
        param_change = param_change.sqrt();

        let new_f_val = f(&new_params);
        let status = criteria.check(iter + 1, f_val, new_f_val, grad_norm, param_change);

        params = new_params;
        f_val = new_f_val;
        f_history.push(f_val);

        if status != ConvergenceStatus::Running {
            return OptResult { params, f_val, iterations: iter + 1, status, f_history };
        }
    }

    OptResult {
        params,
        f_val,
        iterations: criteria.max_iter,
        status: ConvergenceStatus::MaxIterations,
        f_history,
    }
}

// ── Stochastic Gradient Descent ─────────────────────────────────

/// SGD with optional momentum.
pub fn sgd(
    grad_fn: &dyn Fn(&[f64], usize) -> Vec<f64>,
    loss_fn: &dyn Fn(&[f64]) -> f64,
    initial_params: &[f64],
    n_samples: usize,
    schedule: &LrSchedule,
    momentum: f64,
    epochs: usize,
) -> OptResult {
    let n = initial_params.len();
    let mut params = initial_params.to_vec();
    let mut velocity = vec![0.0; n];
    let mut f_history = vec![loss_fn(&params)];
    let mut global_step = 0;

    for _epoch in 0..epochs {
        for sample_idx in 0..n_samples {
            let grad = grad_fn(&params, sample_idx);
            let lr = schedule.get_lr(global_step);

            for i in 0..n {
                velocity[i] = momentum * velocity[i] + lr * grad[i];
                params[i] -= velocity[i];
            }
            global_step += 1;
        }
        f_history.push(loss_fn(&params));
    }

    let f_val = loss_fn(&params);
    OptResult {
        params,
        f_val,
        iterations: global_step,
        status: ConvergenceStatus::MaxIterations,
        f_history,
    }
}

// ── Adam Optimizer ──────────────────────────────────────────────

/// Adam optimizer state.
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate.
    pub lr: f64,
    /// Exponential decay rate for first moment.
    pub beta1: f64,
    /// Exponential decay rate for second moment.
    pub beta2: f64,
    /// Numerical stability term.
    pub epsilon: f64,
    /// First moment estimates.
    m: Vec<f64>,
    /// Second moment estimates.
    v: Vec<f64>,
    /// Time step.
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer.
    pub fn new(n_params: usize, lr: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        }
    }

    /// Create with default hyperparameters (lr=0.001, beta1=0.9, beta2=0.999).
    pub fn with_defaults(n_params: usize) -> Self {
        Self::new(n_params, 0.001, 0.9, 0.999, 1e-8)
    }

    /// Perform one update step given the gradient.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        self.t += 1;
        let t = self.t as f64;

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            let m_hat = self.m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(t));

            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }

    /// Return current time step.
    pub fn time_step(&self) -> usize {
        self.t
    }
}

/// Run Adam optimization to minimize a function.
pub fn adam_optimize(
    f: &dyn Fn(&[f64]) -> f64,
    grad_f: &dyn Fn(&[f64]) -> Vec<f64>,
    initial_params: &[f64],
    lr: f64,
    criteria: &ConvergenceCriteria,
) -> OptResult {
    let mut params = initial_params.to_vec();
    let mut adam = Adam::with_defaults(params.len());
    adam.lr = lr;
    let mut f_val = f(&params);
    let mut f_history = vec![f_val];

    for iter in 0..criteria.max_iter {
        let grad = grad_f(&params);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let old_params = params.clone();

        adam.step(&mut params, &grad);

        let param_change: f64 = params
            .iter()
            .zip(old_params.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        let new_f_val = f(&params);
        let status = criteria.check(iter + 1, f_val, new_f_val, grad_norm, param_change);
        f_val = new_f_val;
        f_history.push(f_val);

        if status != ConvergenceStatus::Running {
            return OptResult { params, f_val, iterations: iter + 1, status, f_history };
        }
    }

    OptResult {
        params,
        f_val,
        iterations: criteria.max_iter,
        status: ConvergenceStatus::MaxIterations,
        f_history,
    }
}

// ── Line Search (Armijo / Backtracking) ─────────────────────────

/// Armijo backtracking line search.
///
/// Returns the step size `alpha` satisfying the Armijo condition:
/// f(x + alpha * d) <= f(x) + c * alpha * (grad . d)
pub fn armijo_line_search(
    f: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    direction: &[f64],
    grad: &[f64],
    c: f64,
    rho: f64,
    max_iter: usize,
) -> f64 {
    let f0 = f(x);
    let slope: f64 = grad.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();
    let mut alpha = 1.0;

    for _ in 0..max_iter {
        let new_x: Vec<f64> = x.iter().zip(direction.iter()).map(|(xi, di)| xi + alpha * di).collect();
        let f_new = f(&new_x);
        if f_new <= f0 + c * alpha * slope {
            return alpha;
        }
        alpha *= rho;
    }
    alpha
}

// ── Newton's Method (1D) ────────────────────────────────────────

/// 1D Newton's method for finding roots of f(x) = 0.
///
/// Returns (root, iterations, converged).
pub fn newton_1d(
    f: &dyn Fn(f64) -> f64,
    f_prime: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, usize, bool) {
    let mut x = x0;
    for iter in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol {
            return (x, iter + 1, true);
        }
        let fpx = f_prime(x);
        if fpx.abs() < f64::EPSILON {
            return (x, iter + 1, false);
        }
        x -= fx / fpx;
    }
    (x, max_iter, false)
}

/// 1D Newton's method for minimizing a function (uses second derivative).
pub fn newton_minimize_1d(
    f_prime: &dyn Fn(f64) -> f64,
    f_double_prime: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, usize, bool) {
    let mut x = x0;
    for iter in 0..max_iter {
        let g = f_prime(x);
        if g.abs() < tol {
            return (x, iter + 1, true);
        }
        let h = f_double_prime(x);
        if h.abs() < f64::EPSILON {
            return (x, iter + 1, false);
        }
        x -= g / h;
    }
    (x, max_iter, false)
}

// ── Golden Section Search ───────────────────────────────────────

/// Golden section search for minimizing a unimodal function on [a, b].
///
/// Returns (x_min, f_min, iterations).
pub fn golden_section_search(
    f: &dyn Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64, usize) {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate

    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    for iter in 0..max_iter {
        if (b - a).abs() < tol {
            let mid = (a + b) / 2.0;
            return (mid, f(mid), iter + 1);
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }
    let mid = (a + b) / 2.0;
    (mid, f(mid), max_iter)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let dy = 200.0 * (x[1] - x[0] * x[0]);
        vec![dx, dy]
    }

    fn quadratic(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn quadratic_grad(x: &[f64]) -> Vec<f64> {
        x.iter().map(|xi| 2.0 * xi).collect()
    }

    #[test]
    fn test_convergence_criteria_max_iter() {
        let c = ConvergenceCriteria { max_iter: 10, ..Default::default() };
        let status = c.check(10, 1.0, 0.9, 1.0, 1.0);
        assert_eq!(status, ConvergenceStatus::MaxIterations);
    }

    #[test]
    fn test_convergence_criteria_value() {
        let c = ConvergenceCriteria { ftol: 0.1, ..Default::default() };
        let status = c.check(1, 1.0, 1.05, 1.0, 1.0);
        assert_eq!(status, ConvergenceStatus::ConvergedValue);
    }

    #[test]
    fn test_convergence_criteria_gradient() {
        let c = ConvergenceCriteria { gtol: 0.1, ftol: 1e-20, ..Default::default() };
        let status = c.check(1, 1.0, 0.5, 0.05, 1.0);
        assert_eq!(status, ConvergenceStatus::ConvergedGradient);
    }

    #[test]
    fn test_lr_constant() {
        let lr = LrSchedule::Constant(0.01);
        assert!((lr.get_lr(0) - 0.01).abs() < 1e-10);
        assert!((lr.get_lr(100) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_lr_step() {
        let lr = LrSchedule::Step { initial_lr: 0.1, decay: 0.5, step_size: 10 };
        assert!((lr.get_lr(0) - 0.1).abs() < 1e-10);
        assert!((lr.get_lr(10) - 0.05).abs() < 1e-10);
        assert!((lr.get_lr(20) - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_lr_exponential() {
        let lr = LrSchedule::Exponential { initial_lr: 0.1, decay: 0.01 };
        assert!((lr.get_lr(0) - 0.1).abs() < 1e-10);
        assert!(lr.get_lr(100) < lr.get_lr(0));
    }

    #[test]
    fn test_lr_cosine() {
        let lr = LrSchedule::Cosine { lr_max: 0.1, lr_min: 0.001, total_steps: 100 };
        assert!((lr.get_lr(0) - 0.1).abs() < 1e-10);
        let mid = lr.get_lr(50);
        assert!(mid > 0.001 && mid < 0.1);
        assert!((lr.get_lr(100) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_lr_display() {
        let lr = LrSchedule::Constant(0.01);
        let s = format!("{}", lr);
        assert!(s.contains("Constant"));
    }

    #[test]
    fn test_gradient_descent_quadratic() {
        let schedule = LrSchedule::Constant(0.1);
        let criteria = ConvergenceCriteria { max_iter: 500, ..Default::default() };
        let result = gradient_descent(&quadratic, &quadratic_grad, &[5.0, 3.0], &schedule, &criteria);
        assert!(result.f_val < 1e-4);
    }

    #[test]
    fn test_numerical_gradient() {
        let grad = numerical_gradient(&quadratic, &[3.0, 4.0], 1e-7);
        assert!((grad[0] - 6.0).abs() < 1e-4);
        assert!((grad[1] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_adam_basic() {
        let criteria = ConvergenceCriteria { max_iter: 2000, ftol: 1e-10, ..Default::default() };
        let result = adam_optimize(&quadratic, &quadratic_grad, &[5.0, 3.0], 0.1, &criteria);
        assert!(result.f_val < 0.1);
    }

    #[test]
    fn test_adam_step() {
        let mut adam = Adam::with_defaults(2);
        let mut params = vec![5.0, 3.0];
        let grad = vec![10.0, 6.0];
        adam.step(&mut params, &grad);
        assert!(params[0] < 5.0);
        assert!(params[1] < 3.0);
        assert_eq!(adam.time_step(), 1);
    }

    #[test]
    fn test_adam_reset() {
        let mut adam = Adam::with_defaults(2);
        let mut params = vec![5.0, 3.0];
        adam.step(&mut params, &[1.0, 1.0]);
        adam.reset();
        assert_eq!(adam.time_step(), 0);
    }

    #[test]
    fn test_armijo_line_search() {
        let x = vec![1.0, 1.0];
        let grad = quadratic_grad(&x);
        let direction: Vec<f64> = grad.iter().map(|g| -g).collect();
        let alpha = armijo_line_search(&quadratic, &x, &direction, &grad, 1e-4, 0.5, 50);
        assert!(alpha > 0.0);
        assert!(alpha <= 1.0);
    }

    #[test]
    fn test_newton_1d_root() {
        // Find root of x^2 - 4 = 0 (should find x = 2 starting from 3)
        let f = |x: f64| x * x - 4.0;
        let fp = |x: f64| 2.0 * x;
        let (root, iters, converged) = newton_1d(&f, &fp, 3.0, 1e-10, 100);
        assert!(converged);
        assert!((root - 2.0).abs() < 1e-8);
        assert!(iters < 20);
    }

    #[test]
    fn test_newton_minimize_1d() {
        // Minimize x^2: f'(x) = 2x, f''(x) = 2
        let fp = |x: f64| 2.0 * x;
        let fpp = |_x: f64| 2.0;
        let (xmin, iters, converged) = newton_minimize_1d(&fp, &fpp, 5.0, 1e-10, 100);
        assert!(converged);
        assert!(xmin.abs() < 1e-8);
        assert!(iters <= 2); // quadratic convergence
    }

    #[test]
    fn test_golden_section_search() {
        let f = |x: f64| (x - 3.0).powi(2);
        let (xmin, fmin, _iters) = golden_section_search(&f, 0.0, 10.0, 1e-8, 100);
        assert!((xmin - 3.0).abs() < 1e-6);
        assert!(fmin < 1e-10);
    }

    #[test]
    fn test_golden_section_search_shifted() {
        let f = |x: f64| (x + 2.0).powi(2) + 1.0;
        let (xmin, fmin, _) = golden_section_search(&f, -10.0, 10.0, 1e-8, 200);
        assert!((xmin - (-2.0)).abs() < 1e-6);
        assert!((fmin - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_basic() {
        let n_samples = 4;
        let grad_fn = |params: &[f64], _idx: usize| -> Vec<f64> {
            vec![2.0 * params[0], 2.0 * params[1]]
        };
        let loss_fn = |params: &[f64]| -> f64 {
            params[0] * params[0] + params[1] * params[1]
        };
        let schedule = LrSchedule::Constant(0.01);
        let result = sgd(&grad_fn, &loss_fn, &[5.0, 3.0], n_samples, &schedule, 0.0, 100);
        assert!(result.f_val < 1.0);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let grad_fn = |params: &[f64], _idx: usize| -> Vec<f64> {
            vec![2.0 * params[0]]
        };
        let loss_fn = |params: &[f64]| -> f64 { params[0] * params[0] };
        let schedule = LrSchedule::Constant(0.01);
        let result = sgd(&grad_fn, &loss_fn, &[5.0], 1, &schedule, 0.9, 200);
        assert!(result.f_val < 0.5);
    }

    #[test]
    fn test_gradient_descent_history() {
        let schedule = LrSchedule::Constant(0.1);
        let criteria = ConvergenceCriteria { max_iter: 10, ..Default::default() };
        let result = gradient_descent(&quadratic, &quadratic_grad, &[5.0], &schedule, &criteria);
        assert!(!result.f_history.is_empty());
        // Function values should generally decrease
        assert!(result.f_history.last().unwrap() < result.f_history.first().unwrap());
    }

    #[test]
    fn test_rosenbrock_adam() {
        let criteria = ConvergenceCriteria { max_iter: 5000, ftol: 1e-12, gtol: 1e-8, ..Default::default() };
        let result = adam_optimize(&rosenbrock, &rosenbrock_grad, &[0.0, 0.0], 0.01, &criteria);
        // Adam should make progress toward (1, 1) minimum
        assert!(result.f_val < rosenbrock(&[0.0, 0.0]));
    }
}
