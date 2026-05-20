//! Bayesian optimization with Gaussian Process surrogate, RBF kernel,
//! acquisition functions (EI, PI, UCB), Cholesky-based GP prediction,
//! sequential model-based optimization loop, and initial random sampling.

// ── Simple deterministic PRNG ────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Kernel ───────────────────────────────────────────────────────

/// Kernel function for the Gaussian Process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// RBF (squared exponential) kernel: σ² × exp(-||x-x'||² / (2l²))
    Rbf { length_scale: f64, variance: f64 },
    /// Matérn 5/2 kernel.
    Matern52 { length_scale: f64, variance: f64 },
}

impl Kernel {
    /// Evaluate the kernel between two points.
    pub fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        match self {
            Self::Rbf { length_scale, variance } => {
                let sq_dist: f64 = x1.iter().zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2)).sum();
                variance * (-sq_dist / (2.0 * length_scale * length_scale)).exp()
            }
            Self::Matern52 { length_scale, variance } => {
                let dist: f64 = x1.iter().zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                let r = (5.0_f64).sqrt() * dist / length_scale;
                variance * (1.0 + r + r * r / 3.0) * (-r).exp()
            }
        }
    }
}

// ── Cholesky decomposition ───────────────────────────────────────

/// Cholesky decomposition of a positive-definite matrix.
/// Returns lower triangular matrix L such that A = L × L^T.
fn cholesky(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = matrix[i][i] - sum;
                if diag <= 0.0 {
                    // Add jitter for numerical stability
                    l[i][j] = 1e-10_f64.sqrt();
                } else {
                    l[i][j] = diag.sqrt();
                }
            } else {
                l[i][j] = (matrix[i][j] - sum) / l[j][j].max(1e-15);
            }
        }
    }

    Some(l)
}

/// Solve L × x = b for x (forward substitution).
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i].max(1e-15);
    }
    x
}

/// Solve L^T × x = b for x (backward substitution).
fn backward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += l[j][i] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i].max(1e-15);
    }
    x
}

// ── Gaussian Process ─────────────────────────────────────────────

/// Gaussian Process surrogate model.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    pub kernel: Kernel,
    pub noise_variance: f64,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    /// Cholesky factor of the kernel matrix + noise.
    chol_l: Option<Vec<Vec<f64>>>,
    /// L^{-1} y (alpha vector for predictions).
    alpha: Vec<f64>,
}

impl GaussianProcess {
    pub fn new(kernel: Kernel, noise_variance: f64) -> Self {
        Self {
            kernel,
            noise_variance,
            x_train: Vec::new(),
            y_train: Vec::new(),
            chol_l: None,
            alpha: Vec::new(),
        }
    }

    /// Add an observation.
    pub fn add_observation(&mut self, x: Vec<f64>, y: f64) {
        self.x_train.push(x);
        self.y_train.push(y);
        self.chol_l = None; // Invalidate cache
    }

    /// Add multiple observations.
    pub fn add_observations(&mut self, xs: Vec<Vec<f64>>, ys: Vec<f64>) {
        for (x, y) in xs.into_iter().zip(ys.into_iter()) {
            self.x_train.push(x);
            self.y_train.push(y);
        }
        self.chol_l = None;
    }

    /// Number of training points.
    pub fn num_observations(&self) -> usize {
        self.x_train.len()
    }

    /// Fit the GP (compute Cholesky and alpha vector).
    pub fn fit(&mut self) {
        let n = self.x_train.len();
        if n == 0 { return; }

        // Build kernel (Gram) matrix
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let kij = self.kernel.evaluate(&self.x_train[i], &self.x_train[j]);
                k[i][j] = kij;
                k[j][i] = kij;
            }
            k[i][i] += self.noise_variance; // Add noise on diagonal
        }

        // Cholesky decomposition
        let l = cholesky(&k).expect("Cholesky failed");
        let z = forward_solve(&l, &self.y_train);
        self.alpha = backward_solve(&l, &z);
        self.chol_l = Some(l);
    }

    /// Predict mean and variance at a new point.
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        let n = self.x_train.len();
        if n == 0 {
            return (0.0, self.kernel.evaluate(x, x) + self.noise_variance);
        }

        let l = match &self.chol_l {
            Some(l) => l,
            None => return (0.0, self.kernel.evaluate(x, x) + self.noise_variance),
        };

        // k* vector
        let k_star: Vec<f64> = (0..n).map(|i| self.kernel.evaluate(x, &self.x_train[i])).collect();

        // Mean: k*^T × alpha
        let mean: f64 = k_star.iter().zip(self.alpha.iter()).map(|(k, a)| k * a).sum();

        // Variance: k(x,x) - v^T v where v = L^{-1} k*
        let v = forward_solve(l, &k_star);
        let var_reduction: f64 = v.iter().map(|vi| vi * vi).sum();
        let k_xx = self.kernel.evaluate(x, x);
        let variance = (k_xx - var_reduction).max(1e-10);

        (mean, variance)
    }

    /// Predict mean only (slightly cheaper).
    pub fn predict_mean(&self, x: &[f64]) -> f64 {
        self.predict(x).0
    }

    /// Get the best observed y value (minimum for minimization).
    pub fn best_observed(&self) -> f64 {
        self.y_train.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Get all training x values.
    pub fn x_train(&self) -> &[Vec<f64>] {
        &self.x_train
    }

    /// Get all training y values.
    pub fn y_train(&self) -> &[f64] {
        &self.y_train
    }
}

// ── Acquisition functions ────────────────────────────────────────

/// Acquisition function type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Acquisition {
    /// Expected Improvement.
    ExpectedImprovement,
    /// Probability of Improvement.
    ProbabilityOfImprovement,
    /// Upper Confidence Bound (with exploration weight κ).
    Ucb(f64),
}

/// Standard normal CDF approximation.
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal PDF.
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Error function approximation (Abramowitz & Stegun).
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
            + t * (1.421413741
                + t * (-1.453152027
                    + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Evaluate acquisition function at a point given GP prediction.
pub fn acquisition_value(acq: Acquisition, mean: f64, variance: f64, best_y: f64) -> f64 {
    let sigma = variance.sqrt();
    if sigma < 1e-15 {
        return 0.0;
    }

    match acq {
        Acquisition::ExpectedImprovement => {
            let improvement = best_y - mean; // minimization
            let z = improvement / sigma;
            improvement * norm_cdf(z) + sigma * norm_pdf(z)
        }
        Acquisition::ProbabilityOfImprovement => {
            let z = (best_y - mean) / sigma;
            norm_cdf(z)
        }
        Acquisition::Ucb(kappa) => {
            // For minimization: lower is better, so negate
            -(mean - kappa * sigma)
        }
    }
}

// ── Bayesian Optimization ────────────────────────────────────────

/// Configuration for Bayesian Optimization.
#[derive(Debug, Clone)]
pub struct BoConfig {
    pub dimensions: usize,
    pub bounds: Vec<(f64, f64)>,
    pub kernel: Kernel,
    pub noise_variance: f64,
    pub acquisition: Acquisition,
    pub initial_samples: usize,
    pub max_iterations: usize,
    pub acquisition_samples: usize,
    pub seed: u64,
}

impl Default for BoConfig {
    fn default() -> Self {
        Self {
            dimensions: 2,
            bounds: vec![(-5.0, 5.0); 2],
            kernel: Kernel::Rbf { length_scale: 1.0, variance: 1.0 },
            noise_variance: 1e-4,
            acquisition: Acquisition::ExpectedImprovement,
            initial_samples: 5,
            max_iterations: 20,
            acquisition_samples: 100,
            seed: 42,
        }
    }
}

/// Result of Bayesian Optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct BoResult {
    pub best_x: Vec<f64>,
    pub best_y: f64,
    pub observations_x: Vec<Vec<f64>>,
    pub observations_y: Vec<f64>,
    pub iterations_run: usize,
}

/// Bayesian Optimization engine.
pub struct BayesianOptimizer {
    config: BoConfig,
    gp: GaussianProcess,
    rng: Rng,
    iteration: usize,
}

impl BayesianOptimizer {
    pub fn new(config: BoConfig) -> Self {
        let gp = GaussianProcess::new(config.kernel, config.noise_variance);
        Self {
            rng: Rng::new(config.seed),
            gp,
            iteration: 0,
            config,
        }
    }

    fn random_point(&mut self) -> Vec<f64> {
        self.config.bounds.iter().map(|&(lo, hi)| {
            lo + self.rng.next_f64() * (hi - lo)
        }).collect()
    }

    /// Perform initial random sampling.
    pub fn initial_sampling<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) {
        for _ in 0..self.config.initial_samples {
            let x = self.random_point();
            let y = obj_fn(&x);
            self.gp.add_observation(x, y);
        }
        self.gp.fit();
    }

    /// Select the next point by optimizing the acquisition function.
    fn select_next(&mut self) -> Vec<f64> {
        let best_y = self.gp.best_observed();
        let mut best_acq = f64::NEG_INFINITY;
        let mut best_x = self.random_point();

        for _ in 0..self.config.acquisition_samples {
            let x = self.random_point();
            let (mean, var) = self.gp.predict(&x);
            let acq = acquisition_value(self.config.acquisition, mean, var, best_y);
            if acq > best_acq {
                best_acq = acq;
                best_x = x;
            }
        }

        best_x
    }

    /// Run one iteration of BO.
    pub fn step<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) {
        let x = self.select_next();
        let y = obj_fn(&x);
        self.gp.add_observation(x, y);
        self.gp.fit();
        self.iteration += 1;
    }

    /// Run the full Bayesian Optimization loop.
    pub fn run<F: Fn(&[f64]) -> f64>(&mut self, obj_fn: &F) -> BoResult {
        self.initial_sampling(obj_fn);

        for _ in 0..self.config.max_iterations {
            self.step(obj_fn);
        }

        let best_idx = self.gp.y_train().iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        BoResult {
            best_x: self.gp.x_train()[best_idx].clone(),
            best_y: self.gp.y_train()[best_idx],
            observations_x: self.gp.x_train().to_vec(),
            observations_y: self.gp.y_train().to_vec(),
            iterations_run: self.iteration,
        }
    }

    /// Get the GP model.
    pub fn gp(&self) -> &GaussianProcess { &self.gp }

    /// Get current iteration.
    pub fn iteration(&self) -> usize { self.iteration }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn branin(x: &[f64]) -> f64 {
        let x1 = x[0];
        let x2 = x[1];
        let a = 1.0;
        let b = 5.1 / (4.0 * std::f64::consts::PI * std::f64::consts::PI);
        let c = 5.0 / std::f64::consts::PI;
        let r = 6.0;
        let s = 10.0;
        let t = 1.0 / (8.0 * std::f64::consts::PI);
        a * (x2 - b * x1 * x1 + c * x1 - r).powi(2) + s * (1.0 - t) * x2.cos() + s
    }

    #[test]
    fn test_rbf_kernel_same_point() {
        let k = Kernel::Rbf { length_scale: 1.0, variance: 1.0 };
        let x = vec![1.0, 2.0];
        assert!((k.evaluate(&x, &x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_different_points() {
        let k = Kernel::Rbf { length_scale: 1.0, variance: 1.0 };
        let x1 = vec![0.0, 0.0];
        let x2 = vec![1.0, 0.0];
        let val = k.evaluate(&x1, &x2);
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_rbf_kernel_symmetry() {
        let k = Kernel::Rbf { length_scale: 1.0, variance: 1.0 };
        let x1 = vec![1.0, 2.0];
        let x2 = vec![3.0, 4.0];
        assert!((k.evaluate(&x1, &x2) - k.evaluate(&x2, &x1)).abs() < 1e-15);
    }

    #[test]
    fn test_matern52_same_point() {
        let k = Kernel::Matern52 { length_scale: 1.0, variance: 1.0 };
        let x = vec![1.0, 2.0];
        assert!((k.evaluate(&x, &x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_2x2() {
        let m = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let l = cholesky(&m).unwrap();
        assert!((l[0][0] - 2.0).abs() < 1e-10);
        assert!((l[1][0] - 1.0).abs() < 1e-10);
        assert!((l[1][1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_gp_no_data() {
        let gp = GaussianProcess::new(Kernel::Rbf { length_scale: 1.0, variance: 1.0 }, 1e-4);
        let (mean, var) = gp.predict(&[0.0, 0.0]);
        assert!((mean - 0.0).abs() < 1e-10);
        assert!(var > 0.0);
    }

    #[test]
    fn test_gp_single_observation() {
        let mut gp = GaussianProcess::new(Kernel::Rbf { length_scale: 1.0, variance: 1.0 }, 1e-4);
        gp.add_observation(vec![0.0], 5.0);
        gp.fit();
        let (mean, var) = gp.predict(&[0.0]);
        assert!((mean - 5.0).abs() < 0.1);
        assert!(var < 0.01); // Very confident at training point
    }

    #[test]
    fn test_gp_uncertainty_increases_with_distance() {
        let mut gp = GaussianProcess::new(Kernel::Rbf { length_scale: 1.0, variance: 1.0 }, 1e-4);
        gp.add_observation(vec![0.0], 1.0);
        gp.fit();
        let (_, var_near) = gp.predict(&[0.1]);
        let (_, var_far) = gp.predict(&[5.0]);
        assert!(var_far > var_near);
    }

    #[test]
    fn test_gp_multiple_observations() {
        let mut gp = GaussianProcess::new(Kernel::Rbf { length_scale: 1.0, variance: 1e-4 }, 1e-4);
        gp.add_observation(vec![0.0], 0.0);
        gp.add_observation(vec![1.0], 1.0);
        gp.add_observation(vec![2.0], 4.0);
        gp.fit();
        assert_eq!(gp.num_observations(), 3);
        let (mean, _) = gp.predict(&[1.0]);
        assert!((mean - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_gp_best_observed() {
        let mut gp = GaussianProcess::new(Kernel::Rbf { length_scale: 1.0, variance: 1.0 }, 1e-4);
        gp.add_observation(vec![0.0], 5.0);
        gp.add_observation(vec![1.0], 2.0);
        gp.add_observation(vec![2.0], 8.0);
        assert!((gp.best_observed() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ei_at_best() {
        // EI should be low at a point equal to best
        let ei = acquisition_value(Acquisition::ExpectedImprovement, 2.0, 0.01, 2.0);
        assert!(ei < 0.1);
    }

    #[test]
    fn test_ei_improves_with_low_mean() {
        let ei_good = acquisition_value(Acquisition::ExpectedImprovement, 1.0, 1.0, 3.0);
        let ei_bad = acquisition_value(Acquisition::ExpectedImprovement, 5.0, 1.0, 3.0);
        assert!(ei_good > ei_bad);
    }

    #[test]
    fn test_pi_range() {
        let pi = acquisition_value(Acquisition::ProbabilityOfImprovement, 2.0, 1.0, 3.0);
        assert!(pi >= 0.0 && pi <= 1.0);
    }

    #[test]
    fn test_ucb_high_uncertainty_preferred() {
        let ucb_high_var = acquisition_value(Acquisition::Ucb(2.0), 3.0, 4.0, 5.0);
        let ucb_low_var = acquisition_value(Acquisition::Ucb(2.0), 3.0, 0.01, 5.0);
        assert!(ucb_high_var > ucb_low_var);
    }

    #[test]
    fn test_norm_cdf_endpoints() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-4);
        assert!(norm_cdf(5.0) > 0.999);
        assert!(norm_cdf(-5.0) < 0.001);
    }

    #[test]
    fn test_norm_pdf_peak() {
        let peak = norm_pdf(0.0);
        assert!((peak - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_bo_initial_sampling() {
        let config = BoConfig {
            dimensions: 1,
            bounds: vec![(-5.0, 5.0)],
            initial_samples: 3,
            max_iterations: 0,
            ..Default::default()
        };
        let mut bo = BayesianOptimizer::new(config);
        bo.initial_sampling(&sphere);
        assert_eq!(bo.gp().num_observations(), 3);
    }

    #[test]
    fn test_bo_step() {
        let config = BoConfig {
            dimensions: 1,
            bounds: vec![(-5.0, 5.0)],
            initial_samples: 3,
            ..Default::default()
        };
        let mut bo = BayesianOptimizer::new(config);
        bo.initial_sampling(&sphere);
        bo.step(&sphere);
        assert_eq!(bo.gp().num_observations(), 4);
        assert_eq!(bo.iteration(), 1);
    }

    #[test]
    fn test_bo_run_sphere() {
        let config = BoConfig {
            dimensions: 1,
            bounds: vec![(-5.0, 5.0)],
            initial_samples: 5,
            max_iterations: 15,
            acquisition_samples: 50,
            ..Default::default()
        };
        let mut bo = BayesianOptimizer::new(config);
        let result = bo.run(&sphere);
        assert!(result.best_y < 5.0, "BO sphere: {}", result.best_y);
        assert_eq!(result.observations_x.len(), 20); // 5 + 15
    }

    #[test]
    fn test_bo_pi_acquisition() {
        let config = BoConfig {
            dimensions: 1,
            bounds: vec![(-3.0, 3.0)],
            acquisition: Acquisition::ProbabilityOfImprovement,
            initial_samples: 5,
            max_iterations: 10,
            ..Default::default()
        };
        let mut bo = BayesianOptimizer::new(config);
        let result = bo.run(&sphere);
        assert!(result.best_y < 10.0);
    }

    #[test]
    fn test_bo_ucb_acquisition() {
        let config = BoConfig {
            dimensions: 1,
            bounds: vec![(-3.0, 3.0)],
            acquisition: Acquisition::Ucb(2.0),
            initial_samples: 5,
            max_iterations: 10,
            ..Default::default()
        };
        let mut bo = BayesianOptimizer::new(config);
        let result = bo.run(&sphere);
        assert!(result.best_y < 10.0);
    }

    #[test]
    fn test_bo_2d() {
        let config = BoConfig {
            dimensions: 2,
            bounds: vec![(-3.0, 3.0), (-3.0, 3.0)],
            initial_samples: 5,
            max_iterations: 15,
            acquisition_samples: 50,
            ..Default::default()
        };
        let mut bo = BayesianOptimizer::new(config);
        let result = bo.run(&sphere);
        assert!(result.best_y < 10.0, "2D BO sphere: {}", result.best_y);
    }

    #[test]
    fn test_bo_matern52_kernel() {
        let config = BoConfig {
            dimensions: 1,
            bounds: vec![(-3.0, 3.0)],
            kernel: Kernel::Matern52 { length_scale: 1.0, variance: 1.0 },
            initial_samples: 5,
            max_iterations: 10,
            ..Default::default()
        };
        let mut bo = BayesianOptimizer::new(config);
        let result = bo.run(&sphere);
        assert!(result.best_y < 10.0);
    }

    #[test]
    fn test_bo_result_structure() {
        let config = BoConfig {
            dimensions: 1,
            bounds: vec![(-5.0, 5.0)],
            initial_samples: 3,
            max_iterations: 5,
            ..Default::default()
        };
        let mut bo = BayesianOptimizer::new(config);
        let result = bo.run(&sphere);
        assert_eq!(result.iterations_run, 5);
        assert_eq!(result.best_x.len(), 1);
        assert_eq!(result.observations_x.len(), 8);
        assert_eq!(result.observations_y.len(), 8);
    }

    #[test]
    fn test_default_config() {
        let c = BoConfig::default();
        assert_eq!(c.dimensions, 2);
        assert_eq!(c.initial_samples, 5);
        assert_eq!(c.max_iterations, 20);
    }
}
