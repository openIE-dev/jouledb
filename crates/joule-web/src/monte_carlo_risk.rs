//! Monte Carlo simulation for financial risk analysis.
//!
//! Provides path generation and variance reduction for risk estimation:
//!
//! - [`GbmPath`] — Geometric Brownian Motion path generator
//! - [`CorrelatedPaths`] — multi-asset paths via Cholesky decomposition
//! - [`AntitheticVariates`] — variance reduction by mirroring paths
//! - [`ControlVariates`] — variance reduction via correlated controls
//! - [`ConvergenceDiag`] — convergence diagnostics for MC estimates

use std::fmt;

// ── Pseudo-RNG ──────────────────────────────────────────────────

/// Simple xoshiro256** PRNG for reproducible simulations.
#[derive(Debug, Clone)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to initialize state
        let mut s = seed;
        let mut state = [0u64; 4];
        for slot in &mut state {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller transform.
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-15 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }
}

// ── GBM Path Generator ─────────────────────────────────────────

/// Geometric Brownian Motion path configuration.
#[derive(Debug, Clone)]
pub struct GbmConfig {
    pub drift: f64,
    pub volatility: f64,
    pub dt: f64,
    pub steps: usize,
    pub seed: u64,
}

impl GbmConfig {
    pub fn new() -> Self {
        Self {
            drift: 0.05,
            volatility: 0.20,
            dt: 1.0 / 252.0,
            steps: 252,
            seed: 42,
        }
    }

    pub fn with_drift(mut self, mu: f64) -> Self {
        self.drift = mu;
        self
    }

    pub fn with_volatility(mut self, sigma: f64) -> Self {
        self.volatility = sigma.abs();
        self
    }

    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt.max(1e-10);
        self
    }

    pub fn with_steps(mut self, s: usize) -> Self {
        self.steps = s.max(1);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for GbmConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Single-asset GBM path generator.
#[derive(Debug)]
pub struct GbmPath {
    config: GbmConfig,
    rng: Rng,
}

impl GbmPath {
    pub fn new(config: GbmConfig) -> Self {
        let rng = Rng::new(config.seed);
        Self { config, rng }
    }

    /// Generate one price path starting from `s0`.
    pub fn generate(&mut self, s0: f64) -> Vec<f64> {
        let mut path = Vec::with_capacity(self.config.steps + 1);
        path.push(s0);
        let dt = self.config.dt;
        let mu = self.config.drift;
        let sigma = self.config.volatility;
        let drift_term = (mu - 0.5 * sigma * sigma) * dt;
        let vol_term = sigma * dt.sqrt();

        for _ in 0..self.config.steps {
            let z = self.rng.next_normal();
            let prev = *path.last().unwrap();
            let next = prev * (drift_term + vol_term * z).exp();
            path.push(next);
        }
        path
    }

    /// Generate n paths, return terminal values.
    pub fn generate_terminals(&mut self, s0: f64, n_paths: usize) -> Vec<f64> {
        (0..n_paths)
            .map(|_| {
                let path = self.generate(s0);
                *path.last().unwrap()
            })
            .collect()
    }

    /// Generate n full paths.
    pub fn generate_paths(&mut self, s0: f64, n_paths: usize) -> Vec<Vec<f64>> {
        (0..n_paths).map(|_| self.generate(s0)).collect()
    }
}

impl fmt::Display for GbmPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GBM(mu={:.2}%, vol={:.2}%, steps={})",
            self.config.drift * 100.0,
            self.config.volatility * 100.0,
            self.config.steps,
        )
    }
}

// ── Cholesky Decomposition ──────────────────────────────────────

/// Cholesky decomposition of a symmetric positive-definite matrix.
pub fn cholesky_decompose(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut lower = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += lower[i][k] * lower[j][k];
            }
            if i == j {
                let diag = matrix[i][i] - sum;
                if diag < 0.0 {
                    return None; // Not positive definite
                }
                lower[i][j] = diag.sqrt();
            } else {
                if lower[j][j].abs() < 1e-15 {
                    return None;
                }
                lower[i][j] = (matrix[i][j] - sum) / lower[j][j];
            }
        }
    }
    Some(lower)
}

// ── Correlated Multi-Asset Paths ────────────────────────────────

/// Multi-asset correlated path generator using Cholesky factorization.
#[derive(Debug)]
pub struct CorrelatedPaths {
    configs: Vec<GbmConfig>,
    cholesky_lower: Vec<Vec<f64>>,
    rng: Rng,
}

impl CorrelatedPaths {
    /// Create from per-asset configs and a correlation matrix.
    pub fn new(configs: Vec<GbmConfig>, correlation: &[Vec<f64>]) -> Option<Self> {
        let cholesky_lower = cholesky_decompose(correlation)?;
        let seed = configs.first().map(|c| c.seed).unwrap_or(42);
        Some(Self {
            configs,
            cholesky_lower,
            rng: Rng::new(seed),
        })
    }

    /// Generate one set of correlated paths.
    pub fn generate(&mut self, initial_prices: &[f64]) -> Vec<Vec<f64>> {
        let n_assets = self.configs.len();
        let steps = self.configs.first().map(|c| c.steps).unwrap_or(252);
        let dt = self.configs.first().map(|c| c.dt).unwrap_or(1.0 / 252.0);

        let mut paths: Vec<Vec<f64>> = initial_prices
            .iter()
            .map(|s0| {
                let mut p = Vec::with_capacity(steps + 1);
                p.push(*s0);
                p
            })
            .collect();

        for _ in 0..steps {
            // Generate independent normals
            let z_indep: Vec<f64> = (0..n_assets).map(|_| self.rng.next_normal()).collect();
            // Correlate via Cholesky
            let mut z_corr = vec![0.0; n_assets];
            for i in 0..n_assets {
                for j in 0..=i {
                    z_corr[i] += self.cholesky_lower[i][j] * z_indep[j];
                }
            }
            // Evolve each asset
            for i in 0..n_assets {
                let mu = self.configs[i].drift;
                let sigma = self.configs[i].volatility;
                let prev = *paths[i].last().unwrap();
                let drift_term = (mu - 0.5 * sigma * sigma) * dt;
                let vol_term = sigma * dt.sqrt();
                paths[i].push(prev * (drift_term + vol_term * z_corr[i]).exp());
            }
        }
        paths
    }

    /// Generate terminal values for n simulations.
    pub fn generate_terminals(&mut self, initial_prices: &[f64], n_sims: usize) -> Vec<Vec<f64>> {
        let n_assets = self.configs.len();
        let mut terminals = vec![Vec::with_capacity(n_sims); n_assets];
        for _ in 0..n_sims {
            let paths = self.generate(initial_prices);
            for (i, path) in paths.iter().enumerate() {
                terminals[i].push(*path.last().unwrap());
            }
        }
        terminals
    }
}

impl fmt::Display for CorrelatedPaths {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CorrelatedPaths({} assets)", self.configs.len())
    }
}

// ── Antithetic Variates ─────────────────────────────────────────

/// Antithetic variates variance reduction for MC simulation.
#[derive(Debug)]
pub struct AntitheticVariates {
    config: GbmConfig,
    rng: Rng,
}

impl AntitheticVariates {
    pub fn new(config: GbmConfig) -> Self {
        let rng = Rng::new(config.seed);
        Self { config, rng }
    }

    /// Generate paired (original, antithetic) terminal values.
    pub fn generate_pairs(&mut self, s0: f64, n_pairs: usize) -> Vec<(f64, f64)> {
        let dt = self.config.dt;
        let mu = self.config.drift;
        let sigma = self.config.volatility;
        let drift_term = (mu - 0.5 * sigma * sigma) * dt;
        let vol_term = sigma * dt.sqrt();

        let mut pairs = Vec::with_capacity(n_pairs);
        for _ in 0..n_pairs {
            let mut s_pos = s0;
            let mut s_neg = s0;
            for _ in 0..self.config.steps {
                let z = self.rng.next_normal();
                s_pos *= (drift_term + vol_term * z).exp();
                s_neg *= (drift_term - vol_term * z).exp();
            }
            pairs.push((s_pos, s_neg));
        }
        pairs
    }

    /// Estimate E[f(S_T)] using antithetic variates.
    pub fn estimate(
        &mut self,
        s0: f64,
        n_pairs: usize,
        payoff: &dyn Fn(f64) -> f64,
    ) -> McEstimate {
        let pairs = self.generate_pairs(s0, n_pairs);
        let estimates: Vec<f64> = pairs
            .iter()
            .map(|(sp, sn)| 0.5 * (payoff(*sp) + payoff(*sn)))
            .collect();
        let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
        let var = if estimates.len() > 1 {
            let ss: f64 = estimates.iter().map(|x| (x - mean) * (x - mean)).sum();
            ss / (estimates.len() - 1) as f64
        } else {
            0.0
        };
        McEstimate {
            mean,
            std_error: (var / estimates.len() as f64).sqrt(),
            n_samples: n_pairs * 2,
        }
    }
}

impl fmt::Display for AntitheticVariates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AntitheticVariates(steps={})", self.config.steps)
    }
}

// ── Control Variates ────────────────────────────────────────────

/// Control variates variance reduction.
pub struct ControlVariates {
    config: GbmConfig,
    rng: Rng,
}

impl ControlVariates {
    pub fn new(config: GbmConfig) -> Self {
        let rng = Rng::new(config.seed);
        Self { config, rng }
    }

    /// Estimate using geometric average as control variate.
    /// `payoff` maps terminal price to payoff.
    pub fn estimate(
        &mut self,
        s0: f64,
        n_paths: usize,
        payoff: &dyn Fn(f64) -> f64,
    ) -> McEstimate {
        let dt = self.config.dt;
        let mu = self.config.drift;
        let sigma = self.config.volatility;
        let drift_term = (mu - 0.5 * sigma * sigma) * dt;
        let vol_term = sigma * dt.sqrt();

        let mut payoffs = Vec::with_capacity(n_paths);
        let mut controls = Vec::with_capacity(n_paths);

        for _ in 0..n_paths {
            let mut s = s0;
            let mut log_sum = 0.0;
            for _ in 0..self.config.steps {
                let z = self.rng.next_normal();
                s *= (drift_term + vol_term * z).exp();
                log_sum += s.ln();
            }
            let geo_avg = (log_sum / self.config.steps as f64).exp();
            payoffs.push(payoff(s));
            controls.push(geo_avg);
        }

        // Compute control variate adjustment
        let n = payoffs.len() as f64;
        let mean_y = payoffs.iter().sum::<f64>() / n;
        let mean_c = controls.iter().sum::<f64>() / n;

        let mut cov_yc = 0.0;
        let mut var_c = 0.0;
        for i in 0..payoffs.len() {
            let dy = payoffs[i] - mean_y;
            let dc = controls[i] - mean_c;
            cov_yc += dy * dc;
            var_c += dc * dc;
        }

        let beta = if var_c > 1e-15 { cov_yc / var_c } else { 0.0 };

        let adjusted: Vec<f64> = payoffs
            .iter()
            .zip(controls.iter())
            .map(|(y, c)| y - beta * (c - mean_c))
            .collect();

        let adj_mean = adjusted.iter().sum::<f64>() / n;
        let adj_var = if adjusted.len() > 1 {
            let ss: f64 = adjusted.iter().map(|x| (x - adj_mean) * (x - adj_mean)).sum();
            ss / (adjusted.len() - 1) as f64
        } else {
            0.0
        };

        McEstimate {
            mean: adj_mean,
            std_error: (adj_var / n).sqrt(),
            n_samples: n_paths,
        }
    }
}

impl fmt::Display for ControlVariates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ControlVariates(steps={})", self.config.steps)
    }
}

// ── MC Estimate ─────────────────────────────────────────────────

/// Result of a Monte Carlo estimation.
#[derive(Debug, Clone)]
pub struct McEstimate {
    pub mean: f64,
    pub std_error: f64,
    pub n_samples: usize,
}

impl McEstimate {
    /// 95% confidence interval.
    pub fn confidence_interval_95(&self) -> (f64, f64) {
        (self.mean - 1.96 * self.std_error, self.mean + 1.96 * self.std_error)
    }

    /// Relative error (coefficient of variation of the mean).
    pub fn relative_error(&self) -> f64 {
        if self.mean.abs() < 1e-15 {
            return f64::INFINITY;
        }
        (self.std_error / self.mean.abs()) * 100.0
    }
}

impl fmt::Display for McEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.confidence_interval_95();
        write!(
            f,
            "MC(mean={:.4}, SE={:.4}, 95%CI=[{:.4}, {:.4}], n={})",
            self.mean, self.std_error, lo, hi, self.n_samples,
        )
    }
}

// ── Convergence Diagnostics ─────────────────────────────────────

/// Monitor convergence of MC estimates as sample size grows.
#[derive(Debug)]
pub struct ConvergenceDiag {
    estimates: Vec<McEstimate>,
    sample_sizes: Vec<usize>,
}

impl ConvergenceDiag {
    pub fn new() -> Self {
        Self {
            estimates: Vec::new(),
            sample_sizes: Vec::new(),
        }
    }

    /// Record an estimate at a given sample size.
    pub fn record(&mut self, n: usize, estimate: McEstimate) {
        self.sample_sizes.push(n);
        self.estimates.push(estimate);
    }

    /// Check if the sequence has converged within tolerance.
    pub fn has_converged(&self, tolerance: f64) -> bool {
        if self.estimates.len() < 2 {
            return false;
        }
        let last = &self.estimates[self.estimates.len() - 1];
        let prev = &self.estimates[self.estimates.len() - 2];
        let diff = (last.mean - prev.mean).abs();
        diff < tolerance && last.std_error < tolerance
    }

    /// Rate of convergence (empirical): log(SE_n / SE_1) / log(n_1 / n_n).
    pub fn convergence_rate(&self) -> f64 {
        if self.estimates.len() < 2 {
            return 0.0;
        }
        let first = &self.estimates[0];
        let last = &self.estimates[self.estimates.len() - 1];
        let n_first = self.sample_sizes[0] as f64;
        let n_last = *self.sample_sizes.last().unwrap() as f64;
        if first.std_error < 1e-15 || n_first < 1.0 || n_last <= n_first {
            return 0.0;
        }
        (first.std_error / last.std_error.max(1e-15)).ln()
            / (n_last / n_first).ln()
    }

    /// Number of recorded checkpoints.
    pub fn checkpoint_count(&self) -> usize {
        self.estimates.len()
    }

    /// Latest estimate.
    pub fn latest(&self) -> Option<&McEstimate> {
        self.estimates.last()
    }
}

impl Default for ConvergenceDiag {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConvergenceDiag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConvergenceDiag({} checkpoints, rate={:.2})",
            self.checkpoint_count(),
            self.convergence_rate(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbm_config_builder() {
        let cfg = GbmConfig::new()
            .with_drift(0.08)
            .with_volatility(0.25)
            .with_steps(100)
            .with_seed(123);
        assert!((cfg.drift - 0.08).abs() < 1e-10);
        assert_eq!(cfg.steps, 100);
    }

    #[test]
    fn test_gbm_path_length() {
        let cfg = GbmConfig::new().with_steps(50);
        let mut gbm = GbmPath::new(cfg);
        let path = gbm.generate(100.0);
        assert_eq!(path.len(), 51, "Should have steps+1 points");
    }

    #[test]
    fn test_gbm_positive_prices() {
        let cfg = GbmConfig::new().with_steps(252);
        let mut gbm = GbmPath::new(cfg);
        let path = gbm.generate(100.0);
        assert!(path.iter().all(|p| *p > 0.0), "GBM prices must be positive");
    }

    #[test]
    fn test_gbm_reproducible() {
        let cfg = GbmConfig::new().with_seed(42).with_steps(100);
        let mut g1 = GbmPath::new(cfg.clone());
        let mut g2 = GbmPath::new(cfg);
        let p1 = g1.generate(100.0);
        let p2 = g2.generate(100.0);
        assert!((p1.last().unwrap() - p2.last().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_gbm_terminals() {
        let cfg = GbmConfig::new().with_steps(50);
        let mut gbm = GbmPath::new(cfg);
        let terms = gbm.generate_terminals(100.0, 100);
        assert_eq!(terms.len(), 100);
        assert!(terms.iter().all(|t| *t > 0.0));
    }

    #[test]
    fn test_cholesky_identity() {
        let id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky_decompose(&id).unwrap();
        assert!((l[0][0] - 1.0).abs() < 1e-10);
        assert!((l[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_non_pd() {
        let bad = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // Not positive definite
        assert!(cholesky_decompose(&bad).is_none());
    }

    #[test]
    fn test_cholesky_correlation() {
        let corr = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];
        let l = cholesky_decompose(&corr);
        assert!(l.is_some(), "Valid correlation matrix should decompose");
    }

    #[test]
    fn test_correlated_paths() {
        let configs = vec![
            GbmConfig::new().with_drift(0.05).with_volatility(0.20),
            GbmConfig::new().with_drift(0.03).with_volatility(0.15),
        ];
        let corr = vec![vec![1.0, 0.6], vec![0.6, 1.0]];
        let mut cp = CorrelatedPaths::new(configs, &corr).unwrap();
        let paths = cp.generate(&[100.0, 50.0]);
        assert_eq!(paths.len(), 2);
        assert!(paths[0].len() > 1);
    }

    #[test]
    fn test_correlated_terminals() {
        let configs = vec![
            GbmConfig::new().with_steps(50).with_volatility(0.20),
            GbmConfig::new().with_steps(50).with_volatility(0.15),
        ];
        let corr = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let mut cp = CorrelatedPaths::new(configs, &corr).unwrap();
        let terms = cp.generate_terminals(&[100.0, 50.0], 20);
        assert_eq!(terms.len(), 2);
        assert_eq!(terms[0].len(), 20);
    }

    #[test]
    fn test_antithetic_pairs() {
        let cfg = GbmConfig::new().with_steps(50).with_seed(42);
        let mut av = AntitheticVariates::new(cfg);
        let pairs = av.generate_pairs(100.0, 50);
        assert_eq!(pairs.len(), 50);
        for (sp, sn) in &pairs {
            assert!(*sp > 0.0 && *sn > 0.0);
        }
    }

    #[test]
    fn test_antithetic_estimate() {
        let cfg = GbmConfig::new().with_steps(50).with_seed(42);
        let mut av = AntitheticVariates::new(cfg);
        let est = av.estimate(100.0, 200, &|s| s);
        assert!(est.mean > 0.0);
        assert!(est.std_error > 0.0);
    }

    #[test]
    fn test_control_variates_estimate() {
        let cfg = GbmConfig::new().with_steps(50).with_seed(42);
        let mut cv = ControlVariates::new(cfg);
        let est = cv.estimate(100.0, 200, &|s| s);
        assert!(est.mean > 0.0);
    }

    #[test]
    fn test_mc_estimate_ci() {
        let est = McEstimate {
            mean: 10.0,
            std_error: 0.5,
            n_samples: 1000,
        };
        let (lo, hi) = est.confidence_interval_95();
        assert!(lo < 10.0 && hi > 10.0);
    }

    #[test]
    fn test_mc_estimate_relative_error() {
        let est = McEstimate {
            mean: 100.0,
            std_error: 1.0,
            n_samples: 1000,
        };
        assert!((est.relative_error() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convergence_not_converged() {
        let diag = ConvergenceDiag::new();
        assert!(!diag.has_converged(0.01));
    }

    #[test]
    fn test_convergence_recording() {
        let mut diag = ConvergenceDiag::new();
        diag.record(100, McEstimate { mean: 10.5, std_error: 0.5, n_samples: 100 });
        diag.record(1000, McEstimate { mean: 10.1, std_error: 0.05, n_samples: 1000 });
        assert_eq!(diag.checkpoint_count(), 2);
        assert!(diag.convergence_rate() > 0.0);
    }

    #[test]
    fn test_convergence_converged() {
        let mut diag = ConvergenceDiag::new();
        diag.record(100, McEstimate { mean: 10.001, std_error: 0.001, n_samples: 100 });
        diag.record(1000, McEstimate { mean: 10.0005, std_error: 0.0005, n_samples: 1000 });
        assert!(diag.has_converged(0.01));
    }

    #[test]
    fn test_display_impls() {
        let cfg = GbmConfig::new();
        let gbm = GbmPath::new(cfg);
        assert!(format!("{gbm}").contains("GBM"));

        let est = McEstimate { mean: 1.0, std_error: 0.1, n_samples: 100 };
        assert!(format!("{est}").contains("MC"));

        let diag = ConvergenceDiag::new();
        assert!(format!("{diag}").contains("ConvergenceDiag"));
    }
}
