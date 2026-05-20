//! Value-at-Risk (VaR) models for portfolio risk measurement.
//!
//! Provides several VaR methodologies commonly used in financial risk management:
//!
//! - [`HistoricalVaR`] — non-parametric VaR from observed P&L history
//! - [`ParametricVaR`] — assumes normal or Student-t return distributions
//! - [`ConditionalVaR`] — CVaR / Expected Shortfall (tail expectation beyond VaR)
//! - [`PortfolioVaR`] — multi-asset VaR using a correlation matrix
//! - [`VarConfig`] — builder for configuring VaR calculations

use std::fmt;

// ── Configuration ───────────────────────────────────────────────

/// Distribution assumption for parametric VaR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distribution {
    /// Standard Gaussian.
    Normal,
    /// Student-t with specified degrees of freedom.
    StudentT(f64),
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::StudentT(df) => write!(f, "StudentT(df={df:.1})"),
        }
    }
}

/// Builder for VaR calculation parameters.
#[derive(Debug, Clone)]
pub struct VarConfig {
    pub confidence: f64,
    pub horizon_days: u32,
    pub distribution: Distribution,
    pub decay_factor: Option<f64>,
}

impl VarConfig {
    pub fn new() -> Self {
        Self {
            confidence: 0.95,
            horizon_days: 1,
            distribution: Distribution::Normal,
            decay_factor: None,
        }
    }

    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence = c.clamp(0.5, 0.9999);
        self
    }

    pub fn with_horizon(mut self, days: u32) -> Self {
        self.horizon_days = days.max(1);
        self
    }

    pub fn with_distribution(mut self, d: Distribution) -> Self {
        self.distribution = d;
        self
    }

    pub fn with_decay_factor(mut self, lambda: f64) -> Self {
        self.decay_factor = Some(lambda.clamp(0.0, 1.0));
        self
    }
}

impl Default for VarConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for VarConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VarConfig(conf={:.2}%, horizon={}d, dist={})",
            self.confidence * 100.0,
            self.horizon_days,
            self.distribution,
        )
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Approximate inverse normal CDF (Beasley-Springer-Moro algorithm).
fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let val = t - num / den;
    if p < 0.5 { -val } else { val }
}

/// Normal PDF.
fn normal_pdf(x: f64) -> f64 {
    let inv_sqrt_2pi = 0.398_942_280_401_432_7;
    inv_sqrt_2pi * (-0.5 * x * x).exp()
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let s: f64 = data.iter().map(|x| (x - m) * (x - m)).sum();
    s / (data.len() - 1) as f64
}

fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Sort a copy of the slice ascending.
fn sorted(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Linear-interpolation percentile on sorted data.
fn percentile_sorted(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    if sorted_data.len() == 1 {
        return sorted_data[0];
    }
    let idx = p * (sorted_data.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted_data.len() - 1);
    let frac = idx - lo as f64;
    sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac
}

// ── Historical VaR ──────────────────────────────────────────────

/// Non-parametric Value-at-Risk computed from historical P&L observations.
#[derive(Debug, Clone)]
pub struct HistoricalVaR {
    config: VarConfig,
    pnl_history: Vec<f64>,
}

impl HistoricalVaR {
    pub fn new(config: VarConfig) -> Self {
        Self {
            config,
            pnl_history: Vec::new(),
        }
    }

    pub fn with_pnl(mut self, pnl: &[f64]) -> Self {
        self.pnl_history = pnl.to_vec();
        self
    }

    /// Compute historical VaR. Returns a positive loss amount.
    pub fn compute(&self) -> f64 {
        if self.pnl_history.is_empty() {
            return 0.0;
        }
        let s = sorted(&self.pnl_history);
        let alpha = 1.0 - self.config.confidence;
        let raw = percentile_sorted(&s, alpha);
        let scaled = raw * (self.config.horizon_days as f64).sqrt();
        -scaled.min(0.0)
    }

    pub fn sample_count(&self) -> usize {
        self.pnl_history.len()
    }
}

impl fmt::Display for HistoricalVaR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HistoricalVaR(n={}, VaR={:.4})",
            self.pnl_history.len(),
            self.compute()
        )
    }
}

// ── Parametric VaR ──────────────────────────────────────────────

/// Parametric VaR assuming a normal or Student-t distribution.
#[derive(Debug, Clone)]
pub struct ParametricVaR {
    config: VarConfig,
    portfolio_value: f64,
    mean_return: f64,
    volatility: f64,
}

impl ParametricVaR {
    pub fn new(config: VarConfig) -> Self {
        Self {
            config,
            portfolio_value: 1.0,
            mean_return: 0.0,
            volatility: 0.01,
        }
    }

    pub fn with_portfolio_value(mut self, v: f64) -> Self {
        self.portfolio_value = v;
        self
    }

    pub fn with_mean_return(mut self, m: f64) -> Self {
        self.mean_return = m;
        self
    }

    pub fn with_volatility(mut self, v: f64) -> Self {
        self.volatility = v.abs();
        self
    }

    /// Estimate parameters from return series.
    pub fn fit_from_returns(mut self, returns: &[f64]) -> Self {
        self.mean_return = mean(returns);
        self.volatility = std_dev(returns);
        self
    }

    /// Compute parametric VaR (positive loss amount).
    pub fn compute(&self) -> f64 {
        let z = match self.config.distribution {
            Distribution::Normal => inv_normal_cdf(self.config.confidence),
            Distribution::StudentT(df) => {
                // Approximate: scale normal quantile by sqrt(df/(df-2))
                let z_norm = inv_normal_cdf(self.config.confidence);
                if df > 2.0 {
                    z_norm * (df / (df - 2.0)).sqrt()
                } else {
                    z_norm * 2.0
                }
            }
        };
        let horizon_scale = (self.config.horizon_days as f64).sqrt();
        let var_pct = self.mean_return * horizon_scale as f64
            - z * self.volatility * horizon_scale;
        (-var_pct * self.portfolio_value).max(0.0)
    }
}

impl fmt::Display for ParametricVaR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParametricVaR(dist={}, vol={:.4}, VaR={:.2})",
            self.config.distribution,
            self.volatility,
            self.compute()
        )
    }
}

// ── Conditional VaR (CVaR / Expected Shortfall) ─────────────────

/// Conditional Value-at-Risk: expected loss in the tail beyond VaR.
#[derive(Debug, Clone)]
pub struct ConditionalVaR {
    config: VarConfig,
}

impl ConditionalVaR {
    pub fn new(config: VarConfig) -> Self {
        Self { config }
    }

    /// Compute CVaR from historical P&L. Returns positive loss amount.
    pub fn compute_historical(&self, pnl: &[f64]) -> f64 {
        if pnl.is_empty() {
            return 0.0;
        }
        let s = sorted(pnl);
        let alpha = 1.0 - self.config.confidence;
        let cutoff_idx = (alpha * s.len() as f64).ceil() as usize;
        let cutoff_idx = cutoff_idx.max(1).min(s.len());
        let tail: Vec<f64> = s[..cutoff_idx].to_vec();
        let tail_mean = mean(&tail);
        (-tail_mean).max(0.0) * (self.config.horizon_days as f64).sqrt()
    }

    /// Compute CVaR under normal distribution assumption.
    pub fn compute_normal(&self, vol: f64, mean_ret: f64, portfolio_value: f64) -> f64 {
        let alpha = 1.0 - self.config.confidence;
        let z = inv_normal_cdf(alpha);
        let pdf_z = normal_pdf(z);
        let horizon = (self.config.horizon_days as f64).sqrt();
        let cvar_pct = -mean_ret * horizon + vol * horizon * pdf_z / alpha;
        (cvar_pct * portfolio_value).max(0.0)
    }

    /// Ratio of CVaR to VaR (always >= 1 for coherent measures).
    pub fn cvar_var_ratio(&self, pnl: &[f64]) -> f64 {
        let hist_var = HistoricalVaR::new(self.config.clone()).with_pnl(pnl).compute();
        let cvar = self.compute_historical(pnl);
        if hist_var > 1e-15 {
            cvar / hist_var
        } else {
            1.0
        }
    }
}

impl fmt::Display for ConditionalVaR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CVaR(conf={:.2}%)", self.config.confidence * 100.0)
    }
}

// ── Portfolio VaR ───────────────────────────────────────────────

/// Multi-asset portfolio VaR using a covariance/correlation matrix.
#[derive(Debug, Clone)]
pub struct PortfolioVaR {
    config: VarConfig,
    weights: Vec<f64>,
    covariance: Vec<Vec<f64>>,
}

impl PortfolioVaR {
    pub fn new(config: VarConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            covariance: Vec::new(),
        }
    }

    pub fn with_weights(mut self, w: &[f64]) -> Self {
        self.weights = w.to_vec();
        self
    }

    /// Set covariance matrix (row-major Vec<Vec<f64>>).
    pub fn with_covariance(mut self, cov: Vec<Vec<f64>>) -> Self {
        self.covariance = cov;
        self
    }

    /// Build covariance from correlation matrix and volatility vector.
    pub fn with_correlation_and_vols(mut self, corr: &[Vec<f64>], vols: &[f64]) -> Self {
        let n = vols.len();
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = corr[i][j] * vols[i] * vols[j];
            }
        }
        self.covariance = cov;
        self
    }

    /// Portfolio variance = w' * Sigma * w.
    pub fn portfolio_variance(&self) -> f64 {
        let n = self.weights.len();
        if n == 0 || self.covariance.len() != n {
            return 0.0;
        }
        let mut var = 0.0;
        for i in 0..n {
            for j in 0..n {
                var += self.weights[i] * self.weights[j] * self.covariance[i][j];
            }
        }
        var.max(0.0)
    }

    /// Compute portfolio VaR.
    pub fn compute(&self) -> f64 {
        let port_vol = self.portfolio_variance().sqrt();
        let z = inv_normal_cdf(self.config.confidence);
        let horizon = (self.config.horizon_days as f64).sqrt();
        z * port_vol * horizon
    }

    /// Marginal VaR contribution for each asset.
    pub fn marginal_var(&self) -> Vec<f64> {
        let n = self.weights.len();
        if n == 0 || self.covariance.len() != n {
            return Vec::new();
        }
        let port_vol = self.portfolio_variance().sqrt();
        if port_vol < 1e-15 {
            return vec![0.0; n];
        }
        let z = inv_normal_cdf(self.config.confidence);
        let horizon = (self.config.horizon_days as f64).sqrt();
        let mut marginal = vec![0.0; n];
        for i in 0..n {
            let mut sigma_w_i = 0.0;
            for j in 0..n {
                sigma_w_i += self.covariance[i][j] * self.weights[j];
            }
            marginal[i] = z * horizon * sigma_w_i / port_vol;
        }
        marginal
    }

    /// Component VaR for each asset (marginal * weight).
    pub fn component_var(&self) -> Vec<f64> {
        let marginal = self.marginal_var();
        marginal
            .iter()
            .zip(self.weights.iter())
            .map(|(m, w)| m * w)
            .collect()
    }

    /// Diversification benefit: sum of individual VaRs minus portfolio VaR.
    pub fn diversification_benefit(&self) -> f64 {
        let n = self.weights.len();
        if n == 0 || self.covariance.len() != n {
            return 0.0;
        }
        let z = inv_normal_cdf(self.config.confidence);
        let horizon = (self.config.horizon_days as f64).sqrt();
        let undiversified: f64 = (0..n)
            .map(|i| self.weights[i].abs() * self.covariance[i][i].sqrt() * z * horizon)
            .sum();
        (undiversified - self.compute()).max(0.0)
    }
}

impl fmt::Display for PortfolioVaR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PortfolioVaR(assets={}, VaR={:.4})",
            self.weights.len(),
            self.compute()
        )
    }
}

// ── VaR Report ──────────────────────────────────────────────────

/// Summary report combining multiple VaR measures.
#[derive(Debug, Clone)]
pub struct VarReport {
    pub historical_var: f64,
    pub parametric_var: f64,
    pub cvar: f64,
    pub confidence: f64,
    pub horizon_days: u32,
    pub sample_count: usize,
}

impl fmt::Display for VarReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VaR Report [{:.1}% / {}d] — Historical: {:.4}, Parametric: {:.4}, CVaR: {:.4} (n={})",
            self.confidence * 100.0,
            self.horizon_days,
            self.historical_var,
            self.parametric_var,
            self.cvar,
            self.sample_count,
        )
    }
}

/// Generate a complete VaR report from P&L data.
pub fn generate_var_report(pnl: &[f64], config: &VarConfig) -> VarReport {
    let hist = HistoricalVaR::new(config.clone()).with_pnl(pnl).compute();
    let vol = std_dev(pnl);
    let m = mean(pnl);
    let param = ParametricVaR::new(config.clone())
        .with_volatility(vol)
        .with_mean_return(m)
        .compute();
    let cvar = ConditionalVaR::new(config.clone()).compute_historical(pnl);
    VarReport {
        historical_var: hist,
        parametric_var: param,
        cvar,
        confidence: config.confidence,
        horizon_days: config.horizon_days,
        sample_count: pnl.len(),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pnl() -> Vec<f64> {
        vec![
            -0.03, 0.01, -0.02, 0.005, -0.015, 0.02, -0.01, 0.008,
            -0.025, 0.012, -0.005, 0.015, -0.018, 0.003, -0.022,
            0.007, -0.012, 0.009, -0.028, 0.004, -0.008, 0.011,
            -0.014, 0.006, -0.019,
        ]
    }

    #[test]
    fn test_var_config_builder() {
        let cfg = VarConfig::new()
            .with_confidence(0.99)
            .with_horizon(10)
            .with_distribution(Distribution::StudentT(5.0));
        assert!((cfg.confidence - 0.99).abs() < 1e-10);
        assert_eq!(cfg.horizon_days, 10);
    }

    #[test]
    fn test_var_config_clamp() {
        let cfg = VarConfig::new().with_confidence(1.5);
        assert!(cfg.confidence <= 0.9999);
    }

    #[test]
    fn test_historical_var_positive() {
        let pnl = sample_pnl();
        let var = HistoricalVaR::new(VarConfig::new().with_confidence(0.95))
            .with_pnl(&pnl)
            .compute();
        assert!(var >= 0.0, "VaR should be non-negative: {var}");
    }

    #[test]
    fn test_historical_var_empty() {
        let var = HistoricalVaR::new(VarConfig::new()).with_pnl(&[]).compute();
        assert!((var - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_historical_var_monotonic_confidence() {
        let pnl = sample_pnl();
        let var95 = HistoricalVaR::new(VarConfig::new().with_confidence(0.95))
            .with_pnl(&pnl)
            .compute();
        let var99 = HistoricalVaR::new(VarConfig::new().with_confidence(0.99))
            .with_pnl(&pnl)
            .compute();
        assert!(var99 >= var95, "99% VaR should >= 95% VaR");
    }

    #[test]
    fn test_parametric_var_normal() {
        let pvar = ParametricVaR::new(VarConfig::new().with_confidence(0.95))
            .with_volatility(0.02)
            .with_portfolio_value(1_000_000.0)
            .compute();
        assert!(pvar > 0.0, "Parametric VaR should be positive: {pvar}");
    }

    #[test]
    fn test_parametric_var_student_t() {
        let cfg = VarConfig::new()
            .with_confidence(0.95)
            .with_distribution(Distribution::StudentT(5.0));
        let normal_var = ParametricVaR::new(VarConfig::new().with_confidence(0.95))
            .with_volatility(0.02)
            .compute();
        let t_var = ParametricVaR::new(cfg)
            .with_volatility(0.02)
            .compute();
        assert!(t_var > normal_var, "Student-t VaR should exceed Normal VaR");
    }

    #[test]
    fn test_parametric_var_fit() {
        let returns = sample_pnl();
        let pvar = ParametricVaR::new(VarConfig::new())
            .fit_from_returns(&returns)
            .compute();
        assert!(pvar >= 0.0);
    }

    #[test]
    fn test_cvar_exceeds_var() {
        let pnl = sample_pnl();
        let cfg = VarConfig::new().with_confidence(0.95);
        let var = HistoricalVaR::new(cfg.clone()).with_pnl(&pnl).compute();
        let cvar = ConditionalVaR::new(cfg).compute_historical(&pnl);
        assert!(cvar >= var - 1e-10, "CVaR should >= VaR: cvar={cvar}, var={var}");
    }

    #[test]
    fn test_cvar_normal() {
        let cfg = VarConfig::new().with_confidence(0.95);
        let cvar = ConditionalVaR::new(cfg).compute_normal(0.02, 0.0, 1_000_000.0);
        assert!(cvar > 0.0);
    }

    #[test]
    fn test_cvar_var_ratio() {
        let pnl = sample_pnl();
        let ratio = ConditionalVaR::new(VarConfig::new().with_confidence(0.95))
            .cvar_var_ratio(&pnl);
        assert!(ratio >= 1.0, "CVaR/VaR ratio should be >= 1: {ratio}");
    }

    #[test]
    fn test_portfolio_var_single_asset() {
        let cfg = VarConfig::new().with_confidence(0.95);
        let pvar = PortfolioVaR::new(cfg)
            .with_weights(&[1.0])
            .with_covariance(vec![vec![0.0004]]);
        let var = pvar.compute();
        assert!(var > 0.0, "Single-asset portfolio VaR should be positive");
    }

    #[test]
    fn test_portfolio_var_diversification() {
        let cfg = VarConfig::new().with_confidence(0.95);
        let corr = vec![
            vec![1.0, 0.3],
            vec![0.3, 1.0],
        ];
        let vols = vec![0.02, 0.03];
        let pvar = PortfolioVaR::new(cfg)
            .with_weights(&[0.5, 0.5])
            .with_correlation_and_vols(&corr, &vols);
        let benefit = pvar.diversification_benefit();
        assert!(benefit > 0.0, "Should have diversification benefit: {benefit}");
    }

    #[test]
    fn test_marginal_var_sums_to_portfolio() {
        let cfg = VarConfig::new().with_confidence(0.95);
        let pvar = PortfolioVaR::new(cfg)
            .with_weights(&[0.6, 0.4])
            .with_covariance(vec![
                vec![0.0004, 0.0001],
                vec![0.0001, 0.0009],
            ]);
        let comp_var: f64 = pvar.component_var().iter().sum();
        let total = pvar.compute();
        assert!(
            (comp_var - total).abs() < 1e-10,
            "Component VaRs should sum to portfolio VaR"
        );
    }

    #[test]
    fn test_portfolio_var_empty() {
        let pvar = PortfolioVaR::new(VarConfig::new());
        assert!((pvar.compute() - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_var_report() {
        let pnl = sample_pnl();
        let cfg = VarConfig::new().with_confidence(0.95);
        let report = generate_var_report(&pnl, &cfg);
        assert!(report.historical_var >= 0.0);
        assert!(report.parametric_var >= 0.0);
        assert!(report.cvar >= 0.0);
        assert_eq!(report.sample_count, pnl.len());
    }

    #[test]
    fn test_inv_normal_cdf_symmetry() {
        let z95 = inv_normal_cdf(0.95);
        let z05 = inv_normal_cdf(0.05);
        assert!((z95 + z05).abs() < 0.05, "Should be approximately symmetric");
    }

    #[test]
    fn test_horizon_scaling() {
        let pnl = sample_pnl();
        let var1 = HistoricalVaR::new(VarConfig::new().with_horizon(1))
            .with_pnl(&pnl)
            .compute();
        let var10 = HistoricalVaR::new(VarConfig::new().with_horizon(10))
            .with_pnl(&pnl)
            .compute();
        assert!(var10 > var1, "10-day VaR should exceed 1-day VaR");
    }

    #[test]
    fn test_display_impls() {
        let cfg = VarConfig::new().with_confidence(0.99);
        let s = format!("{cfg}");
        assert!(s.contains("99.00%"));

        let hist = HistoricalVaR::new(VarConfig::new()).with_pnl(&[1.0, -1.0]);
        let s2 = format!("{hist}");
        assert!(s2.contains("n=2"));
    }

    #[test]
    fn test_distribution_display() {
        assert_eq!(format!("{}", Distribution::Normal), "Normal");
        let t = Distribution::StudentT(4.0);
        assert!(format!("{t}").contains("4.0"));
    }
}
