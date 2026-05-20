//! Asset allocation — strategic allocation, tactical tilts, Black-Litterman
//! model, risk-based allocation, constant-mix rebalance, and glide path for
//! target-date portfolios.
//!
//! Pure Rust, std-only. All floating-point values are `f64`.

use std::fmt;

// ── AssetClass ──────────────────────────────────────────────────

/// An asset class with a target allocation and risk properties.
#[derive(Debug, Clone)]
pub struct AssetClass {
    pub name: String,
    pub target_weight: f64,
    pub expected_return: f64,
    pub volatility: f64,
}

impl AssetClass {
    pub fn new(name: &str, target_weight: f64, expected_return: f64, volatility: f64) -> Self {
        Self {
            name: name.to_string(),
            target_weight,
            expected_return,
            volatility,
        }
    }
}

impl fmt::Display for AssetClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AssetClass({}, target={:.1}%, ret={:.2}%)",
            self.name,
            self.target_weight * 100.0,
            self.expected_return * 100.0
        )
    }
}

// ── StrategicAllocation ─────────────────────────────────────────

/// Long-horizon strategic asset allocation policy.
#[derive(Debug, Clone)]
pub struct StrategicAllocation {
    pub classes: Vec<AssetClass>,
}

impl StrategicAllocation {
    pub fn new(classes: Vec<AssetClass>) -> Self {
        Self { classes }
    }

    /// Sum of all target weights (should be 1.0 for a valid allocation).
    pub fn total_weight(&self) -> f64 {
        self.classes.iter().map(|c| c.target_weight).sum()
    }

    /// Validate that all weights are non-negative and sum to 1.
    pub fn is_valid(&self) -> bool {
        let sum = self.total_weight();
        (sum - 1.0).abs() < 1e-8 && self.classes.iter().all(|c| c.target_weight >= 0.0)
    }

    /// Expected portfolio return under strategic weights.
    pub fn expected_return(&self) -> f64 {
        self.classes
            .iter()
            .map(|c| c.target_weight * c.expected_return)
            .sum()
    }

    /// Portfolio volatility assuming zero correlation (lower bound estimate).
    pub fn volatility_uncorrelated(&self) -> f64 {
        let var: f64 = self
            .classes
            .iter()
            .map(|c| c.target_weight * c.target_weight * c.volatility * c.volatility)
            .sum();
        var.sqrt()
    }

    /// Retrieve target weight for a class by name.
    pub fn weight_for(&self, name: &str) -> Option<f64> {
        self.classes
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.target_weight)
    }
}

impl fmt::Display for StrategicAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StrategicAllocation({} classes)", self.classes.len())
    }
}

// ── TacticalTilt ────────────────────────────────────────────────

/// A short-term overweight / underweight relative to strategic targets.
#[derive(Debug, Clone)]
pub struct TacticalTilt {
    pub class_name: String,
    pub tilt_bps: f64,
    pub confidence: f64,
    pub horizon_days: u32,
}

impl TacticalTilt {
    pub fn new(class_name: &str, tilt_bps: f64, confidence: f64, horizon_days: u32) -> Self {
        Self {
            class_name: class_name.to_string(),
            tilt_bps,
            confidence: confidence.clamp(0.0, 1.0),
            horizon_days,
        }
    }

    /// Tilt as a decimal fraction.
    pub fn tilt_decimal(&self) -> f64 {
        self.tilt_bps / 10_000.0
    }
}

impl fmt::Display for TacticalTilt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TacticalTilt({}, {:+.0}bps, conf={:.0}%)",
            self.class_name,
            self.tilt_bps,
            self.confidence * 100.0
        )
    }
}

/// Apply tactical tilts to a strategic allocation, keeping weights normalised.
pub fn apply_tactical_tilts(
    strategic: &StrategicAllocation,
    tilts: &[TacticalTilt],
) -> Vec<f64> {
    let mut weights: Vec<f64> = strategic.classes.iter().map(|c| c.target_weight).collect();
    for tilt in tilts {
        if let Some(idx) = strategic
            .classes
            .iter()
            .position(|c| c.name == tilt.class_name)
        {
            weights[idx] += tilt.tilt_decimal() * tilt.confidence;
        }
    }
    // Clip to zero and re-normalise
    for w in weights.iter_mut() {
        if *w < 0.0 {
            *w = 0.0;
        }
    }
    let sum: f64 = weights.iter().sum();
    if sum > 1e-15 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    weights
}

// ── Black-Litterman ─────────────────────────────────────────────

/// Black-Litterman model parameters.
#[derive(Debug, Clone)]
pub struct BlackLitterman {
    pub tau: f64,
    pub risk_aversion: f64,
    pub market_weights: Vec<f64>,
    pub covariance: Vec<f64>,
    pub n_assets: usize,
}

impl BlackLitterman {
    pub fn new(
        market_weights: Vec<f64>,
        covariance: Vec<f64>,
        n_assets: usize,
    ) -> Self {
        Self {
            tau: 0.05,
            risk_aversion: 2.5,
            market_weights,
            covariance,
            n_assets,
        }
    }

    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    pub fn with_risk_aversion(mut self, ra: f64) -> Self {
        self.risk_aversion = ra;
        self
    }

    /// Implied equilibrium excess returns: pi = delta * Sigma * w_mkt.
    pub fn equilibrium_returns(&self) -> Vec<f64> {
        let n = self.n_assets;
        let mut pi = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                pi[i] += self.risk_aversion * self.covariance[i * n + j] * self.market_weights[j];
            }
        }
        pi
    }

    /// Posterior expected returns given a single absolute view:
    /// asset `view_asset` will return `view_return` with confidence `view_confidence` (0..1).
    /// Returns the BL posterior mean vector.
    pub fn posterior_returns(
        &self,
        view_asset: usize,
        view_return: f64,
        view_confidence: f64,
    ) -> Vec<f64> {
        let n = self.n_assets;
        let pi = self.equilibrium_returns();
        if view_asset >= n || view_confidence <= 0.0 {
            return pi;
        }
        // Omega scalar for this single-view case
        let omega = self.tau * self.covariance[view_asset * n + view_asset] / view_confidence;
        // Simplified BL for single absolute view
        let tau_sigma_p = self.tau * self.covariance[view_asset * n + view_asset];
        let scaling = tau_sigma_p / (tau_sigma_p + omega);

        let mut posterior = pi.clone();
        let view_diff = view_return - pi[view_asset];
        for i in 0..n {
            let cov_ip = self.tau * self.covariance[i * n + view_asset];
            posterior[i] += (cov_ip / (tau_sigma_p + omega)) * view_diff;
        }
        let _ = scaling; // used implicitly via the formula
        posterior
    }
}

impl fmt::Display for BlackLitterman {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlackLitterman(n={}, tau={:.3}, delta={:.2})",
            self.n_assets, self.tau, self.risk_aversion
        )
    }
}

// ── Risk-Based Allocation ───────────────────────────────────────

/// Inverse-volatility weighting: allocate inversely proportional to volatility.
pub fn inverse_volatility_weights(volatilities: &[f64]) -> Vec<f64> {
    let inv: Vec<f64> = volatilities
        .iter()
        .map(|v| if *v > 1e-15 { 1.0 / v } else { 0.0 })
        .collect();
    let total: f64 = inv.iter().sum();
    if total < 1e-15 {
        return vec![1.0 / volatilities.len() as f64; volatilities.len()];
    }
    inv.iter().map(|x| x / total).collect()
}

/// Equal-risk contribution (approximate): each asset contributes equal risk.
/// Uses iterative proportional fitting.
pub fn equal_risk_contribution(
    covariance: &[f64],
    n_assets: usize,
    max_iter: usize,
) -> Vec<f64> {
    let mut weights = vec![1.0 / n_assets as f64; n_assets];
    let lr = 0.01;

    for _ in 0..max_iter {
        let port_var = portfolio_var(&weights, covariance, n_assets);
        if port_var < 1e-18 {
            break;
        }
        let port_vol = port_var.sqrt();
        let target_rc = port_vol / n_assets as f64;

        let mut grad = vec![0.0; n_assets];
        for i in 0..n_assets {
            let mut sigma_w_i = 0.0;
            for j in 0..n_assets {
                sigma_w_i += covariance[i * n_assets + j] * weights[j];
            }
            let rc_i = weights[i] * sigma_w_i / port_vol;
            grad[i] = rc_i - target_rc;
        }
        for i in 0..n_assets {
            weights[i] -= lr * grad[i];
            if weights[i] < 1e-10 {
                weights[i] = 1e-10;
            }
        }
        let s: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= s;
        }
    }
    weights
}

fn portfolio_var(weights: &[f64], cov: &[f64], n: usize) -> f64 {
    let mut v = 0.0;
    for i in 0..n {
        for j in 0..n {
            v += weights[i] * weights[j] * cov[i * n + j];
        }
    }
    v
}

// ── Constant-Mix Rebalance ──────────────────────────────────────

/// Given current market values and target weights, compute the trades needed
/// to rebalance to the constant-mix targets. Returns the dollar amount to
/// trade for each asset (positive = buy, negative = sell).
pub fn constant_mix_trades(
    current_values: &[f64],
    target_weights: &[f64],
) -> Vec<f64> {
    let total: f64 = current_values.iter().sum();
    current_values
        .iter()
        .zip(target_weights.iter())
        .map(|(val, tw)| tw * total - val)
        .collect()
}

// ── Glide Path ──────────────────────────────────────────────────

/// Glide path configuration for target-date portfolios.
#[derive(Debug, Clone)]
pub struct GlidePath {
    pub start_equity: f64,
    pub end_equity: f64,
    pub years_to_target: u32,
    pub curve_exponent: f64,
}

impl GlidePath {
    pub fn new(years_to_target: u32) -> Self {
        Self {
            start_equity: 0.90,
            end_equity: 0.30,
            years_to_target,
            curve_exponent: 1.0,
        }
    }

    pub fn with_start_equity(mut self, e: f64) -> Self {
        self.start_equity = e.clamp(0.0, 1.0);
        self
    }

    pub fn with_end_equity(mut self, e: f64) -> Self {
        self.end_equity = e.clamp(0.0, 1.0);
        self
    }

    pub fn with_curve_exponent(mut self, exp: f64) -> Self {
        self.curve_exponent = exp.max(0.1);
        self
    }

    /// Equity allocation at a given number of years remaining.
    pub fn equity_at(&self, years_remaining: u32) -> f64 {
        if self.years_to_target == 0 {
            return self.end_equity;
        }
        let t = (years_remaining as f64 / self.years_to_target as f64).clamp(0.0, 1.0);
        let frac = t.powf(self.curve_exponent);
        self.end_equity + (self.start_equity - self.end_equity) * frac
    }

    /// Bond allocation (complement of equity).
    pub fn bond_at(&self, years_remaining: u32) -> f64 {
        1.0 - self.equity_at(years_remaining)
    }

    /// Generate the full glide path as (year, equity_pct) pairs.
    pub fn path(&self) -> Vec<(u32, f64)> {
        (0..=self.years_to_target)
            .rev()
            .map(|y| (self.years_to_target - y, self.equity_at(y)))
            .collect()
    }
}

impl fmt::Display for GlidePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GlidePath({}y, equity {:.0}%→{:.0}%)",
            self.years_to_target,
            self.start_equity * 100.0,
            self.end_equity * 100.0
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn three_class_allocation() -> StrategicAllocation {
        StrategicAllocation::new(vec![
            AssetClass::new("Equity", 0.60, 0.08, 0.16),
            AssetClass::new("Bonds", 0.30, 0.03, 0.05),
            AssetClass::new("Cash", 0.10, 0.01, 0.01),
        ])
    }

    #[test]
    fn test_strategic_valid() {
        let sa = three_class_allocation();
        assert!(sa.is_valid());
    }

    #[test]
    fn test_strategic_expected_return() {
        let sa = three_class_allocation();
        let ret = sa.expected_return();
        let expected = 0.60 * 0.08 + 0.30 * 0.03 + 0.10 * 0.01;
        assert!((ret - expected).abs() < 1e-10);
    }

    #[test]
    fn test_strategic_weight_for() {
        let sa = three_class_allocation();
        assert!((sa.weight_for("Bonds").unwrap() - 0.30).abs() < 1e-10);
        assert!(sa.weight_for("Commodities").is_none());
    }

    #[test]
    fn test_strategic_display() {
        let sa = three_class_allocation();
        let s = format!("{sa}");
        assert!(s.contains("3 classes"));
    }

    #[test]
    fn test_asset_class_display() {
        let ac = AssetClass::new("Equity", 0.6, 0.08, 0.16);
        let s = format!("{ac}");
        assert!(s.contains("Equity"));
    }

    #[test]
    fn test_tactical_tilt_apply() {
        let sa = three_class_allocation();
        let tilts = vec![TacticalTilt::new("Equity", 200.0, 0.8, 30)];
        let w = apply_tactical_tilts(&sa, &tilts);
        assert!(w[0] > 0.60); // equity should be overweight
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tactical_tilt_display() {
        let t = TacticalTilt::new("Equity", 200.0, 0.8, 30);
        let s = format!("{t}");
        assert!(s.contains("200"));
    }

    #[test]
    fn test_black_litterman_equilibrium() {
        let cov = vec![
            0.04, 0.006, 0.006, 0.01,
        ];
        let bl = BlackLitterman::new(vec![0.6, 0.4], cov, 2);
        let pi = bl.equilibrium_returns();
        assert_eq!(pi.len(), 2);
        assert!(pi[0] > 0.0);
    }

    #[test]
    fn test_black_litterman_posterior() {
        let cov = vec![0.04, 0.006, 0.006, 0.01];
        let bl = BlackLitterman::new(vec![0.6, 0.4], cov, 2);
        let pi = bl.equilibrium_returns();
        let posterior = bl.posterior_returns(0, 0.12, 0.9);
        // Posterior for asset 0 should be pulled toward 0.12
        assert!(posterior[0] > pi[0]);
    }

    #[test]
    fn test_black_litterman_display() {
        let bl = BlackLitterman::new(vec![0.5, 0.5], vec![0.04, 0.0, 0.0, 0.04], 2);
        let s = format!("{bl}");
        assert!(s.contains("BlackLitterman"));
    }

    #[test]
    fn test_inverse_volatility() {
        let vols = vec![0.20, 0.10, 0.05];
        let w = inverse_volatility_weights(&vols);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(w[2] > w[1]); // lower vol → higher weight
        assert!(w[1] > w[0]);
    }

    #[test]
    fn test_equal_risk_contribution_sums() {
        let cov = vec![0.04, 0.01, 0.01, 0.02];
        let w = equal_risk_contribution(&cov, 2, 500);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_constant_mix_trades() {
        let vals = vec![6000.0, 3000.0, 1000.0];
        let tgt = vec![0.60, 0.30, 0.10];
        let trades = constant_mix_trades(&vals, &tgt);
        let net: f64 = trades.iter().sum();
        assert!(net.abs() < 1e-8); // zero-sum trades
    }

    #[test]
    fn test_glide_path_equity_at() {
        let gp = GlidePath::new(40);
        let eq_start = gp.equity_at(40);
        let eq_end = gp.equity_at(0);
        assert!((eq_start - 0.90).abs() < 1e-10);
        assert!((eq_end - 0.30).abs() < 1e-10);
    }

    #[test]
    fn test_glide_path_bond_complement() {
        let gp = GlidePath::new(30);
        for y in 0..=30 {
            let eq = gp.equity_at(y);
            let bd = gp.bond_at(y);
            assert!((eq + bd - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_glide_path_monotone() {
        let gp = GlidePath::new(30);
        let path = gp.path();
        for w in path.windows(2) {
            assert!(w[0].1 >= w[1].1); // equity decreases over time
        }
    }

    #[test]
    fn test_glide_path_display() {
        let gp = GlidePath::new(40);
        let s = format!("{gp}");
        assert!(s.contains("40y"));
    }

    #[test]
    fn test_glide_path_custom() {
        let gp = GlidePath::new(20)
            .with_start_equity(0.80)
            .with_end_equity(0.40)
            .with_curve_exponent(2.0);
        assert!((gp.start_equity - 0.80).abs() < 1e-10);
        assert!((gp.end_equity - 0.40).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_uncorrelated() {
        let sa = three_class_allocation();
        let vol = sa.volatility_uncorrelated();
        assert!(vol > 0.0);
    }
}
