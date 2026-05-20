//! Portfolio optimization — mean-variance (Markowitz), minimum variance,
//! maximum Sharpe ratio, efficient frontier sampling, constraint handling
//! (long-only, sector limits), and `PortfolioConfig` builder.
//!
//! Pure Rust, std-only. All floating-point values are `f64`.

use std::fmt;

// ── PortfolioConfig ─────────────────────────────────────────────

/// Builder for configuring a portfolio optimisation run.
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    pub risk_free_rate: f64,
    pub long_only: bool,
    pub max_weight: f64,
    pub min_weight: f64,
    pub sector_limits: Vec<SectorLimit>,
    pub frontier_points: usize,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl PortfolioConfig {
    pub fn new() -> Self {
        Self {
            risk_free_rate: 0.02,
            long_only: true,
            max_weight: 1.0,
            min_weight: 0.0,
            sector_limits: Vec::new(),
            frontier_points: 50,
            tolerance: 1e-8,
            max_iterations: 5000,
        }
    }

    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    pub fn with_long_only(mut self, flag: bool) -> Self {
        self.long_only = flag;
        if flag {
            self.min_weight = 0.0;
        }
        self
    }

    pub fn with_max_weight(mut self, w: f64) -> Self {
        self.max_weight = w.clamp(0.0, 1.0);
        self
    }

    pub fn with_min_weight(mut self, w: f64) -> Self {
        self.min_weight = w;
        self
    }

    pub fn with_sector_limit(mut self, limit: SectorLimit) -> Self {
        self.sector_limits.push(limit);
        self
    }

    pub fn with_frontier_points(mut self, n: usize) -> Self {
        self.frontier_points = n.max(2);
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PortfolioConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PortfolioConfig(rf={:.4}, long_only={}, frontier_pts={})",
            self.risk_free_rate, self.long_only, self.frontier_points
        )
    }
}

// ── SectorLimit ─────────────────────────────────────────────────

/// Constrains the aggregate weight of assets in a given sector.
#[derive(Debug, Clone)]
pub struct SectorLimit {
    pub sector_id: usize,
    pub max_weight: f64,
    pub asset_indices: Vec<usize>,
}

impl SectorLimit {
    pub fn new(sector_id: usize, max_weight: f64, asset_indices: Vec<usize>) -> Self {
        Self {
            sector_id,
            max_weight: max_weight.clamp(0.0, 1.0),
            asset_indices,
        }
    }

    pub fn sector_weight(&self, weights: &[f64]) -> f64 {
        self.asset_indices
            .iter()
            .filter_map(|i| weights.get(*i))
            .sum()
    }

    pub fn is_satisfied(&self, weights: &[f64]) -> bool {
        self.sector_weight(weights) <= self.max_weight + 1e-10
    }
}

impl fmt::Display for SectorLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SectorLimit(id={}, max={:.2}%, assets={})",
            self.sector_id,
            self.max_weight * 100.0,
            self.asset_indices.len()
        )
    }
}

// ── PortfolioResult ─────────────────────────────────────────────

/// Output of an optimisation run.
#[derive(Debug, Clone)]
pub struct PortfolioResult {
    pub weights: Vec<f64>,
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl PortfolioResult {
    pub fn asset_count(&self) -> usize {
        self.weights.len()
    }

    pub fn active_assets(&self) -> usize {
        self.weights.iter().filter(|&&w| w.abs() > 1e-8).count()
    }
}

impl fmt::Display for PortfolioResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Portfolio(ret={:.4}, vol={:.4}, sharpe={:.4}, assets={})",
            self.expected_return,
            self.volatility,
            self.sharpe_ratio,
            self.active_assets()
        )
    }
}

// ── Covariance helpers ──────────────────────────────────────────

/// Compute the sample mean of each column in a returns matrix
/// stored in row-major order (n_periods × n_assets).
pub fn column_means(returns: &[f64], n_assets: usize) -> Vec<f64> {
    let n_periods = returns.len() / n_assets;
    if n_periods == 0 {
        return vec![0.0; n_assets];
    }
    let mut means = vec![0.0; n_assets];
    for row in 0..n_periods {
        for col in 0..n_assets {
            means[col] += returns[row * n_assets + col];
        }
    }
    for m in means.iter_mut() {
        *m /= n_periods as f64;
    }
    means
}

/// Sample covariance matrix (n_assets × n_assets), row-major.
pub fn covariance_matrix(returns: &[f64], n_assets: usize) -> Vec<f64> {
    let n_periods = returns.len() / n_assets;
    let means = column_means(returns, n_assets);
    let mut cov = vec![0.0; n_assets * n_assets];
    if n_periods < 2 {
        return cov;
    }
    for row in 0..n_periods {
        for i in 0..n_assets {
            let di = returns[row * n_assets + i] - means[i];
            for j in i..n_assets {
                let dj = returns[row * n_assets + j] - means[j];
                cov[i * n_assets + j] += di * dj;
            }
        }
    }
    let denom = (n_periods - 1) as f64;
    for i in 0..n_assets {
        for j in i..n_assets {
            cov[i * n_assets + j] /= denom;
            cov[j * n_assets + i] = cov[i * n_assets + j];
        }
    }
    cov
}

/// Portfolio variance given weights and covariance matrix.
pub fn portfolio_variance(weights: &[f64], cov: &[f64], n: usize) -> f64 {
    let mut var = 0.0;
    for i in 0..n {
        for j in 0..n {
            var += weights[i] * weights[j] * cov[i * n + j];
        }
    }
    var
}

/// Portfolio expected return given weights and mean returns.
pub fn portfolio_return(weights: &[f64], means: &[f64]) -> f64 {
    weights.iter().zip(means.iter()).map(|(w, m)| w * m).sum()
}

// ── Constraint projection ───────────────────────────────────────

/// Projects weights onto the simplex (sum = 1, each >= 0) for long-only.
fn project_simplex(weights: &mut [f64]) {
    let n = weights.len();
    let mut sorted: Vec<f64> = weights.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0;
    let mut rho = 0;
    for (i, &v) in sorted.iter().enumerate() {
        cumsum += v;
        if v - (cumsum - 1.0) / (i + 1) as f64 > 0.0 {
            rho = i + 1;
        }
    }
    let theta = (sorted[..rho].iter().sum::<f64>() - 1.0) / rho as f64;
    for w in weights.iter_mut() {
        *w = (*w - theta).max(0.0);
    }
}

/// Enforce constraints from config on a weight vector.
fn apply_constraints(weights: &mut [f64], config: &PortfolioConfig) {
    if config.long_only {
        project_simplex(weights);
    }
    for w in weights.iter_mut() {
        *w = w.clamp(config.min_weight, config.max_weight);
    }
    // Re-normalise to sum = 1
    let s: f64 = weights.iter().sum();
    if s.abs() > 1e-15 {
        for w in weights.iter_mut() {
            *w /= s;
        }
    }
}

// ── Mean-Variance (Markowitz) optimiser ─────────────────────────

/// Run mean-variance optimisation targeting a specific return level.
pub fn mean_variance_optimize(
    returns: &[f64],
    n_assets: usize,
    target_return: f64,
    config: &PortfolioConfig,
) -> PortfolioResult {
    let means = column_means(returns, n_assets);
    let cov = covariance_matrix(returns, n_assets);
    let mut weights = vec![1.0 / n_assets as f64; n_assets];
    let lr = 0.01;

    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        // Gradient of variance w.r.t weights
        let mut grad = vec![0.0; n_assets];
        for i in 0..n_assets {
            for j in 0..n_assets {
                grad[i] += 2.0 * cov[i * n_assets + j] * weights[j];
            }
        }
        // Penalty for deviating from target return
        let current_ret = portfolio_return(&weights, &means);
        let ret_penalty = 2.0 * (current_ret - target_return);
        for i in 0..n_assets {
            grad[i] += ret_penalty * means[i];
        }
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tolerance {
            converged = true;
            break;
        }
        for i in 0..n_assets {
            weights[i] -= lr * grad[i];
        }
        apply_constraints(&mut weights, config);
    }

    let vol = portfolio_variance(&weights, &cov, n_assets).sqrt();
    let ret = portfolio_return(&weights, &means);
    let sharpe = if vol > 1e-15 {
        (ret - config.risk_free_rate) / vol
    } else {
        0.0
    };

    PortfolioResult {
        weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: sharpe,
        iterations,
        converged,
    }
}

// ── Minimum Variance ────────────────────────────────────────────

/// Find the minimum-variance portfolio.
pub fn minimum_variance(
    returns: &[f64],
    n_assets: usize,
    config: &PortfolioConfig,
) -> PortfolioResult {
    let cov = covariance_matrix(returns, n_assets);
    let means = column_means(returns, n_assets);
    let mut weights = vec![1.0 / n_assets as f64; n_assets];
    let lr = 0.01;
    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let mut grad = vec![0.0; n_assets];
        for i in 0..n_assets {
            for j in 0..n_assets {
                grad[i] += 2.0 * cov[i * n_assets + j] * weights[j];
            }
        }
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tolerance {
            converged = true;
            break;
        }
        for i in 0..n_assets {
            weights[i] -= lr * grad[i];
        }
        apply_constraints(&mut weights, config);
    }

    let vol = portfolio_variance(&weights, &cov, n_assets).sqrt();
    let ret = portfolio_return(&weights, &means);
    let sharpe = if vol > 1e-15 {
        (ret - config.risk_free_rate) / vol
    } else {
        0.0
    };

    PortfolioResult {
        weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: sharpe,
        iterations,
        converged,
    }
}

// ── Maximum Sharpe ──────────────────────────────────────────────

/// Find the maximum Sharpe ratio portfolio via gradient ascent on Sharpe.
pub fn maximum_sharpe(
    returns: &[f64],
    n_assets: usize,
    config: &PortfolioConfig,
) -> PortfolioResult {
    let means = column_means(returns, n_assets);
    let cov = covariance_matrix(returns, n_assets);
    let mut weights = vec![1.0 / n_assets as f64; n_assets];
    let lr = 0.005;
    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let ret = portfolio_return(&weights, &means);
        let var = portfolio_variance(&weights, &cov, n_assets);
        let vol = var.sqrt().max(1e-15);
        let excess = ret - config.risk_free_rate;

        // d(Sharpe)/dw_i = (mu_i * vol - excess * (Sigma w)_i / vol) / vol^2
        let mut grad = vec![0.0; n_assets];
        for i in 0..n_assets {
            let mut cov_w_i = 0.0;
            for j in 0..n_assets {
                cov_w_i += cov[i * n_assets + j] * weights[j];
            }
            grad[i] = (means[i] * vol - excess * cov_w_i / vol) / (vol * vol);
        }

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tolerance {
            converged = true;
            break;
        }
        for i in 0..n_assets {
            weights[i] += lr * grad[i]; // ascent
        }
        apply_constraints(&mut weights, config);
    }

    let vol = portfolio_variance(&weights, &cov, n_assets).sqrt();
    let ret = portfolio_return(&weights, &means);
    let sharpe = if vol > 1e-15 {
        (ret - config.risk_free_rate) / vol
    } else {
        0.0
    };

    PortfolioResult {
        weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: sharpe,
        iterations,
        converged,
    }
}

// ── Efficient Frontier ──────────────────────────────────────────

/// A single point on the efficient frontier.
#[derive(Debug, Clone)]
pub struct FrontierPoint {
    pub target_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub weights: Vec<f64>,
}

impl fmt::Display for FrontierPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FrontierPoint(ret={:.4}, vol={:.4}, sharpe={:.4})",
            self.target_return, self.volatility, self.sharpe_ratio
        )
    }
}

/// Sample the efficient frontier at `config.frontier_points` equally-spaced
/// return levels between the min and max single-asset mean returns.
pub fn efficient_frontier(
    returns: &[f64],
    n_assets: usize,
    config: &PortfolioConfig,
) -> Vec<FrontierPoint> {
    let means = column_means(returns, n_assets);
    let min_ret = means.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ret = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n = config.frontier_points;
    let step = if n > 1 {
        (max_ret - min_ret) / (n - 1) as f64
    } else {
        0.0
    };

    (0..n)
        .map(|i| {
            let target = min_ret + step * i as f64;
            let result = mean_variance_optimize(returns, n_assets, target, config);
            FrontierPoint {
                target_return: target,
                volatility: result.volatility,
                sharpe_ratio: result.sharpe_ratio,
                weights: result.weights,
            }
        })
        .collect()
}

// ── Equal-weight baseline ───────────────────────────────────────

/// Return the 1/N equal-weight portfolio as a baseline.
pub fn equal_weight_portfolio(
    returns: &[f64],
    n_assets: usize,
    risk_free_rate: f64,
) -> PortfolioResult {
    let weights = vec![1.0 / n_assets as f64; n_assets];
    let means = column_means(returns, n_assets);
    let cov = covariance_matrix(returns, n_assets);
    let vol = portfolio_variance(&weights, &cov, n_assets).sqrt();
    let ret = portfolio_return(&weights, &means);
    let sharpe = if vol > 1e-15 {
        (ret - risk_free_rate) / vol
    } else {
        0.0
    };
    PortfolioResult {
        weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: sharpe,
        iterations: 0,
        converged: true,
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_returns() -> Vec<f64> {
        // 6 periods × 3 assets (row-major)
        vec![
            0.01, 0.02, 0.015,
            -0.005, 0.01, 0.005,
            0.02, -0.01, 0.01,
            0.005, 0.015, -0.005,
            -0.01, 0.005, 0.02,
            0.015, 0.008, 0.003,
        ]
    }

    #[test]
    fn test_column_means() {
        let r = sample_returns();
        let m = column_means(&r, 3);
        assert_eq!(m.len(), 3);
        let expected = (0.01 - 0.005 + 0.02 + 0.005 - 0.01 + 0.015) / 6.0;
        assert!((m[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_symmetric() {
        let r = sample_returns();
        let cov = covariance_matrix(&r, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((cov[i * 3 + j] - cov[j * 3 + i]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_covariance_positive_diagonal() {
        let r = sample_returns();
        let cov = covariance_matrix(&r, 3);
        for i in 0..3 {
            assert!(cov[i * 3 + i] >= 0.0);
        }
    }

    #[test]
    fn test_portfolio_variance_nonneg() {
        let r = sample_returns();
        let cov = covariance_matrix(&r, 3);
        let w = vec![0.3, 0.4, 0.3];
        assert!(portfolio_variance(&w, &cov, 3) >= 0.0);
    }

    #[test]
    fn test_portfolio_return_equal_weight() {
        let means = vec![0.1, 0.2, 0.3];
        let w = vec![1.0 / 3.0; 3];
        let ret = portfolio_return(&w, &means);
        assert!((ret - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let c = PortfolioConfig::new()
            .with_risk_free_rate(0.03)
            .with_long_only(true)
            .with_max_weight(0.5)
            .with_frontier_points(20);
        assert!((c.risk_free_rate - 0.03).abs() < 1e-15);
        assert!(c.long_only);
        assert!((c.max_weight - 0.5).abs() < 1e-15);
        assert_eq!(c.frontier_points, 20);
    }

    #[test]
    fn test_config_display() {
        let c = PortfolioConfig::new();
        let s = format!("{c}");
        assert!(s.contains("PortfolioConfig"));
    }

    #[test]
    fn test_sector_limit_satisfied() {
        let sl = SectorLimit::new(0, 0.5, vec![0, 1]);
        let w = vec![0.2, 0.2, 0.6];
        assert!(sl.is_satisfied(&w));
    }

    #[test]
    fn test_sector_limit_violated() {
        let sl = SectorLimit::new(0, 0.3, vec![0, 1]);
        let w = vec![0.2, 0.2, 0.6];
        assert!(!sl.is_satisfied(&w));
    }

    #[test]
    fn test_minimum_variance_weights_sum() {
        let r = sample_returns();
        let config = PortfolioConfig::new().with_max_iterations(2000);
        let res = minimum_variance(&r, 3, &config);
        let s: f64 = res.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minimum_variance_long_only() {
        let r = sample_returns();
        let config = PortfolioConfig::new().with_long_only(true);
        let res = minimum_variance(&r, 3, &config);
        for &w in &res.weights {
            assert!(w >= -1e-10);
        }
    }

    #[test]
    fn test_maximum_sharpe_weights_sum() {
        let r = sample_returns();
        let config = PortfolioConfig::new().with_max_iterations(3000);
        let res = maximum_sharpe(&r, 3, &config);
        let s: f64 = res.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_variance_target() {
        let r = sample_returns();
        let config = PortfolioConfig::new().with_max_iterations(3000);
        let _res = mean_variance_optimize(&r, 3, 0.01, &config);
        // Should produce a valid result without panic
    }

    #[test]
    fn test_efficient_frontier_length() {
        let r = sample_returns();
        let config = PortfolioConfig::new()
            .with_frontier_points(10)
            .with_max_iterations(500);
        let frontier = efficient_frontier(&r, 3, &config);
        assert_eq!(frontier.len(), 10);
    }

    #[test]
    fn test_frontier_point_display() {
        let fp = FrontierPoint {
            target_return: 0.05,
            volatility: 0.12,
            sharpe_ratio: 0.25,
            weights: vec![0.5, 0.5],
        };
        let s = format!("{fp}");
        assert!(s.contains("FrontierPoint"));
    }

    #[test]
    fn test_equal_weight_portfolio() {
        let r = sample_returns();
        let res = equal_weight_portfolio(&r, 3, 0.02);
        assert_eq!(res.weights.len(), 3);
        assert!((res.weights[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!(res.converged);
    }

    #[test]
    fn test_portfolio_result_active_assets() {
        let res = PortfolioResult {
            weights: vec![0.5, 0.0, 0.5, 0.0],
            expected_return: 0.1,
            volatility: 0.15,
            sharpe_ratio: 0.53,
            iterations: 100,
            converged: true,
        };
        assert_eq!(res.active_assets(), 2);
        assert_eq!(res.asset_count(), 4);
    }

    #[test]
    fn test_portfolio_result_display() {
        let res = PortfolioResult {
            weights: vec![0.5, 0.5],
            expected_return: 0.1,
            volatility: 0.15,
            sharpe_ratio: 0.53,
            iterations: 100,
            converged: true,
        };
        let s = format!("{res}");
        assert!(s.contains("Portfolio("));
    }

    #[test]
    fn test_project_simplex_already_valid() {
        let mut w = vec![0.3, 0.3, 0.4];
        project_simplex(&mut w);
        let s: f64 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_asset_covariance() {
        let r = vec![0.01, -0.02, 0.03, 0.005];
        let cov = covariance_matrix(&r, 1);
        assert!(cov[0] > 0.0);
    }
}
