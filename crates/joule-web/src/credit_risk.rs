//! Credit risk models for counterparty and portfolio credit analysis.
//!
//! Provides structural and statistical credit risk tools:
//!
//! - [`MertonModel`] — structural model deriving PD from asset dynamics
//! - [`CreditScoring`] — logistic regression credit score model
//! - [`LossEstimator`] — PD/LGD/EAD expected and unexpected loss
//! - [`CreditMigration`] — transition matrix for rating migrations
//! - [`CreditConfig`] — builder for credit risk parameters

use std::fmt;

// ── Configuration ───────────────────────────────────────────────

/// Builder for credit risk calculation parameters.
#[derive(Debug, Clone)]
pub struct CreditConfig {
    pub risk_free_rate: f64,
    pub time_horizon: f64,
    pub confidence_level: f64,
    pub correlation: f64,
}

impl CreditConfig {
    pub fn new() -> Self {
        Self {
            risk_free_rate: 0.05,
            time_horizon: 1.0,
            confidence_level: 0.999,
            correlation: 0.20,
        }
    }

    pub fn with_risk_free_rate(mut self, r: f64) -> Self {
        self.risk_free_rate = r;
        self
    }

    pub fn with_time_horizon(mut self, t: f64) -> Self {
        self.time_horizon = t.max(0.01);
        self
    }

    pub fn with_confidence_level(mut self, c: f64) -> Self {
        self.confidence_level = c.clamp(0.9, 0.9999);
        self
    }

    pub fn with_correlation(mut self, rho: f64) -> Self {
        self.correlation = rho.clamp(0.0, 1.0);
        self
    }
}

impl Default for CreditConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CreditConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CreditConfig(rf={:.2}%, T={:.1}y, conf={:.2}%, rho={:.2})",
            self.risk_free_rate * 100.0,
            self.time_horizon,
            self.confidence_level * 100.0,
            self.correlation,
        )
    }
}

// ── Math Helpers ────────────────────────────────────────────────

/// Approximate standard normal CDF using Abramowitz & Stegun.
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();
    0.5 * (1.0 + sign * y)
}

/// Approximate inverse normal CDF.
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

// ── Merton Structural Model ────────────────────────────────────

/// Merton (1974) structural credit model.
///
/// Models firm default as the event where asset value falls below
/// the face value of debt at maturity.
#[derive(Debug, Clone)]
pub struct MertonModel {
    pub asset_value: f64,
    pub debt_face: f64,
    pub asset_vol: f64,
    pub config: CreditConfig,
}

impl MertonModel {
    pub fn new(asset_value: f64, debt_face: f64, asset_vol: f64) -> Self {
        Self {
            asset_value,
            debt_face,
            asset_vol,
            config: CreditConfig::new(),
        }
    }

    pub fn with_config(mut self, cfg: CreditConfig) -> Self {
        self.config = cfg;
        self
    }

    /// Distance to default (number of std deviations).
    pub fn distance_to_default(&self) -> f64 {
        let r = self.config.risk_free_rate;
        let t = self.config.time_horizon;
        let sigma = self.asset_vol;
        if sigma < 1e-15 || t < 1e-15 {
            return if self.asset_value > self.debt_face {
                f64::INFINITY
            } else {
                0.0
            };
        }
        let num = (self.asset_value / self.debt_face).ln()
            + (r - 0.5 * sigma * sigma) * t;
        num / (sigma * t.sqrt())
    }

    /// Probability of default under risk-neutral measure.
    pub fn default_probability(&self) -> f64 {
        let dd = self.distance_to_default();
        normal_cdf(-dd)
    }

    /// d1 parameter for Black-Scholes-Merton.
    fn d1(&self) -> f64 {
        let t = self.config.time_horizon;
        let sigma = self.asset_vol;
        if sigma < 1e-15 || t < 1e-15 {
            return 0.0;
        }
        let r = self.config.risk_free_rate;
        ((self.asset_value / self.debt_face).ln()
            + (r + 0.5 * sigma * sigma) * t)
            / (sigma * t.sqrt())
    }

    /// d2 parameter for Black-Scholes-Merton.
    fn d2(&self) -> f64 {
        self.d1() - self.asset_vol * self.config.time_horizon.sqrt()
    }

    /// Equity value under Merton model (call option on firm assets).
    pub fn equity_value(&self) -> f64 {
        let r = self.config.risk_free_rate;
        let t = self.config.time_horizon;
        self.asset_value * normal_cdf(self.d1())
            - self.debt_face * (-r * t).exp() * normal_cdf(self.d2())
    }

    /// Debt value under Merton model.
    pub fn debt_value(&self) -> f64 {
        self.asset_value - self.equity_value()
    }

    /// Credit spread implied by the model (annualized).
    pub fn credit_spread(&self) -> f64 {
        let t = self.config.time_horizon;
        let r = self.config.risk_free_rate;
        let debt_val = self.debt_value();
        if debt_val <= 0.0 || t < 1e-15 {
            return 0.0;
        }
        let yield_risky = -(debt_val / self.debt_face).ln() / t;
        (yield_risky - r).max(0.0)
    }
}

impl fmt::Display for MertonModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Merton(A={:.0}, D={:.0}, vol={:.2}%, DD={:.2}, PD={:.4}%)",
            self.asset_value,
            self.debt_face,
            self.asset_vol * 100.0,
            self.distance_to_default(),
            self.default_probability() * 100.0,
        )
    }
}

// ── Credit Scoring ──────────────────────────────────────────────

/// Logistic regression credit scoring model.
#[derive(Debug, Clone)]
pub struct CreditScoring {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub feature_names: Vec<String>,
}

impl CreditScoring {
    pub fn new(intercept: f64, coefficients: &[f64]) -> Self {
        Self {
            intercept,
            coefficients: coefficients.to_vec(),
            feature_names: (0..coefficients.len())
                .map(|i| format!("x{i}"))
                .collect(),
        }
    }

    pub fn with_feature_names(mut self, names: &[&str]) -> Self {
        self.feature_names = names.iter().map(|n| n.to_string()).collect();
        self
    }

    /// Probability of default (logistic output).
    pub fn predict_pd(&self, features: &[f64]) -> f64 {
        let n = features.len().min(self.coefficients.len());
        let mut logit = self.intercept;
        for i in 0..n {
            logit += self.coefficients[i] * features[i];
        }
        1.0 / (1.0 + (-logit).exp())
    }

    /// Credit score mapped to [300, 850] range.
    pub fn credit_score(&self, features: &[f64]) -> f64 {
        let pd = self.predict_pd(features);
        let score = 850.0 - (pd * 550.0);
        score.clamp(300.0, 850.0)
    }

    /// Feature importance (absolute coefficient magnitude).
    pub fn feature_importance(&self) -> Vec<(String, f64)> {
        self.feature_names
            .iter()
            .zip(self.coefficients.iter())
            .map(|(name, coeff)| (name.clone(), coeff.abs()))
            .collect()
    }

    /// Marginal effect of feature i at given values.
    pub fn marginal_effect(&self, features: &[f64], feature_idx: usize) -> f64 {
        if feature_idx >= self.coefficients.len() {
            return 0.0;
        }
        let pd = self.predict_pd(features);
        self.coefficients[feature_idx] * pd * (1.0 - pd)
    }
}

impl fmt::Display for CreditScoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CreditScoring(intercept={:.3}, {} features)",
            self.intercept,
            self.coefficients.len(),
        )
    }
}

// ── PD/LGD/EAD Loss Estimator ──────────────────────────────────

/// Expected and unexpected credit loss estimation.
#[derive(Debug, Clone)]
pub struct LossEstimator {
    pub pd: f64,
    pub lgd: f64,
    pub ead: f64,
    pub config: CreditConfig,
}

impl LossEstimator {
    pub fn new(pd: f64, lgd: f64, ead: f64) -> Self {
        Self {
            pd: pd.clamp(0.0, 1.0),
            lgd: lgd.clamp(0.0, 1.0),
            ead: ead.max(0.0),
            config: CreditConfig::new(),
        }
    }

    pub fn with_config(mut self, cfg: CreditConfig) -> Self {
        self.config = cfg;
        self
    }

    /// Expected loss = PD * LGD * EAD.
    pub fn expected_loss(&self) -> f64 {
        self.pd * self.lgd * self.ead
    }

    /// Loss variance for a single obligor.
    fn loss_variance(&self) -> f64 {
        self.lgd * self.lgd * self.ead * self.ead * self.pd * (1.0 - self.pd)
    }

    /// Unexpected loss (standard deviation of loss).
    pub fn unexpected_loss(&self) -> f64 {
        self.loss_variance().sqrt()
    }

    /// Conditional PD under the Vasicek single-factor model.
    pub fn conditional_pd(&self) -> f64 {
        let rho = self.config.correlation;
        let z = inv_normal_cdf(self.config.confidence_level);
        let pd_inv = inv_normal_cdf(self.pd);
        if rho >= 1.0 {
            return self.pd;
        }
        let arg = (pd_inv + rho.sqrt() * z) / (1.0 - rho).sqrt();
        normal_cdf(arg)
    }

    /// Regulatory/economic capital (Vasicek model).
    pub fn economic_capital(&self) -> f64 {
        let cpd = self.conditional_pd();
        (cpd * self.lgd * self.ead - self.expected_loss()).max(0.0)
    }

    /// Risk-weighted assets (simplified IRB approach).
    pub fn risk_weighted_assets(&self) -> f64 {
        let k = self.economic_capital() / self.ead.max(1e-15);
        k * 12.5 * self.ead
    }
}

impl fmt::Display for LossEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LossEstimator(PD={:.2}%, LGD={:.0}%, EAD={:.0}, EL={:.2})",
            self.pd * 100.0,
            self.lgd * 100.0,
            self.ead,
            self.expected_loss(),
        )
    }
}

// ── Portfolio Loss ──────────────────────────────────────────────

/// Portfolio-level credit loss aggregation.
#[derive(Debug, Clone)]
pub struct PortfolioLoss {
    pub obligors: Vec<LossEstimator>,
}

impl PortfolioLoss {
    pub fn new() -> Self {
        Self {
            obligors: Vec::new(),
        }
    }

    pub fn with_obligor(mut self, est: LossEstimator) -> Self {
        self.obligors.push(est);
        self
    }

    /// Total expected loss across all obligors.
    pub fn total_expected_loss(&self) -> f64 {
        self.obligors.iter().map(|o| o.expected_loss()).sum()
    }

    /// Total unexpected loss (assuming independence — lower bound).
    pub fn total_unexpected_loss_independent(&self) -> f64 {
        let var: f64 = self.obligors.iter().map(|o| o.loss_variance()).sum();
        var.sqrt()
    }

    /// Total economic capital.
    pub fn total_economic_capital(&self) -> f64 {
        self.obligors.iter().map(|o| o.economic_capital()).sum()
    }

    /// Concentration index (Herfindahl on EAD).
    pub fn herfindahl_index(&self) -> f64 {
        let total_ead: f64 = self.obligors.iter().map(|o| o.ead).sum();
        if total_ead < 1e-15 {
            return 0.0;
        }
        self.obligors
            .iter()
            .map(|o| {
                let share = o.ead / total_ead;
                share * share
            })
            .sum()
    }
}

impl Default for PortfolioLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PortfolioLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PortfolioLoss({} obligors, EL={:.2}, HHI={:.4})",
            self.obligors.len(),
            self.total_expected_loss(),
            self.herfindahl_index(),
        )
    }
}

// ── Credit Migration ────────────────────────────────────────────

/// Rating migration/transition matrix.
#[derive(Debug, Clone)]
pub struct CreditMigration {
    pub rating_labels: Vec<String>,
    pub transition_matrix: Vec<Vec<f64>>,
}

impl CreditMigration {
    pub fn new(labels: &[&str], matrix: Vec<Vec<f64>>) -> Self {
        Self {
            rating_labels: labels.iter().map(|s| s.to_string()).collect(),
            transition_matrix: matrix,
        }
    }

    /// Standard S&P-like 7-state transition matrix (simplified).
    pub fn standard_7state() -> Self {
        let labels = vec!["AAA", "AA", "A", "BBB", "BB", "B", "D"];
        let matrix = vec![
            vec![0.9081, 0.0833, 0.0068, 0.0006, 0.0012, 0.0000, 0.0000],
            vec![0.0070, 0.9065, 0.0779, 0.0064, 0.0006, 0.0014, 0.0002],
            vec![0.0009, 0.0227, 0.9105, 0.0552, 0.0074, 0.0026, 0.0007],
            vec![0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0030],
            vec![0.0003, 0.0014, 0.0067, 0.0773, 0.8053, 0.0884, 0.0206],
            vec![0.0000, 0.0011, 0.0024, 0.0043, 0.0648, 0.8346, 0.0928],
            vec![0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        ];
        Self::new(&labels, matrix)
    }

    /// Probability of migrating from one rating to another (1 period).
    pub fn migration_probability(&self, from: usize, to: usize) -> f64 {
        if from < self.transition_matrix.len() && to < self.transition_matrix[from].len() {
            self.transition_matrix[from][to]
        } else {
            0.0
        }
    }

    /// Multi-period transition matrix (matrix exponentiation by squaring).
    pub fn multi_period(&self, periods: u32) -> Vec<Vec<f64>> {
        let n = self.transition_matrix.len();
        if periods == 0 {
            let mut identity = vec![vec![0.0; n]; n];
            for i in 0..n {
                identity[i][i] = 1.0;
            }
            return identity;
        }
        if periods == 1 {
            return self.transition_matrix.clone();
        }
        let mut result = self.transition_matrix.clone();
        for _ in 1..periods {
            result = mat_mul(&result, &self.transition_matrix);
        }
        result
    }

    /// Default probability (transition to last state) from a given rating.
    pub fn default_probability(&self, from: usize) -> f64 {
        let n = self.transition_matrix.len();
        if from < n && n > 0 {
            self.transition_matrix[from][n - 1]
        } else {
            0.0
        }
    }

    /// Expected rating drift (change in rating index, positive = downgrade).
    pub fn expected_drift(&self, from: usize) -> f64 {
        let n = self.transition_matrix.len();
        if from >= n {
            return 0.0;
        }
        let mut drift = 0.0;
        for to in 0..n {
            drift += self.transition_matrix[from][to] * (to as f64 - from as f64);
        }
        drift
    }
}

/// Matrix multiplication for square matrices.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

impl fmt::Display for CreditMigration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CreditMigration({} ratings: {:?})",
            self.rating_labels.len(),
            self.rating_labels,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credit_config_builder() {
        let cfg = CreditConfig::new()
            .with_risk_free_rate(0.03)
            .with_time_horizon(2.0)
            .with_confidence_level(0.999)
            .with_correlation(0.15);
        assert!((cfg.risk_free_rate - 0.03).abs() < 1e-10);
        assert!((cfg.time_horizon - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_merton_healthy_firm() {
        let m = MertonModel::new(150.0, 100.0, 0.20);
        let dd = m.distance_to_default();
        assert!(dd > 0.0, "Healthy firm should have positive DD");
        let pd = m.default_probability();
        assert!(pd < 0.50, "Healthy firm PD should be low: {pd}");
    }

    #[test]
    fn test_merton_distressed_firm() {
        let m = MertonModel::new(80.0, 100.0, 0.40);
        let pd = m.default_probability();
        assert!(pd > 0.10, "Distressed firm should have elevated PD: {pd}");
    }

    #[test]
    fn test_merton_equity_value() {
        let m = MertonModel::new(150.0, 100.0, 0.20);
        let equity = m.equity_value();
        assert!(equity > 0.0, "Equity should be positive");
        assert!(equity < m.asset_value, "Equity < Asset value");
    }

    #[test]
    fn test_merton_credit_spread() {
        let m = MertonModel::new(120.0, 100.0, 0.30);
        let spread = m.credit_spread();
        assert!(spread >= 0.0, "Credit spread should be non-negative");
    }

    #[test]
    fn test_merton_asset_decomposition() {
        let m = MertonModel::new(150.0, 100.0, 0.20);
        let sum = m.equity_value() + m.debt_value();
        assert!(
            (sum - m.asset_value).abs() < 0.01,
            "Equity + Debt should = Asset: {sum} vs {}",
            m.asset_value,
        );
    }

    #[test]
    fn test_credit_scoring_pd() {
        let model = CreditScoring::new(-2.0, &[0.5, -0.3, 0.1]);
        let pd = model.predict_pd(&[1.0, 2.0, 3.0]);
        assert!(pd > 0.0 && pd < 1.0, "PD should be in (0,1): {pd}");
    }

    #[test]
    fn test_credit_score_range() {
        let model = CreditScoring::new(-1.0, &[0.3, -0.2]);
        let score = model.credit_score(&[0.5, 1.0]);
        assert!(score >= 300.0 && score <= 850.0, "Score out of range: {score}");
    }

    #[test]
    fn test_credit_scoring_importance() {
        let model = CreditScoring::new(0.0, &[0.5, -0.8, 0.1])
            .with_feature_names(&["income", "debt_ratio", "age"]);
        let imp = model.feature_importance();
        assert_eq!(imp.len(), 3);
        assert_eq!(imp[1].0, "debt_ratio");
    }

    #[test]
    fn test_marginal_effect() {
        let model = CreditScoring::new(-1.0, &[0.5]);
        let me = model.marginal_effect(&[0.0], 0);
        assert!(me > 0.0, "Positive coeff should give positive ME");
    }

    #[test]
    fn test_expected_loss() {
        let est = LossEstimator::new(0.02, 0.45, 1_000_000.0);
        let el = est.expected_loss();
        let expected = 0.02 * 0.45 * 1_000_000.0;
        assert!((el - expected).abs() < 1e-6);
    }

    #[test]
    fn test_unexpected_loss() {
        let est = LossEstimator::new(0.02, 0.45, 1_000_000.0);
        let ul = est.unexpected_loss();
        assert!(ul > 0.0, "UL should be positive");
        assert!(ul > est.expected_loss(), "UL typically exceeds EL for low PD");
    }

    #[test]
    fn test_economic_capital() {
        let est = LossEstimator::new(0.02, 0.45, 1_000_000.0)
            .with_config(CreditConfig::new().with_confidence_level(0.999));
        let ec = est.economic_capital();
        assert!(ec > 0.0, "EC should be positive");
    }

    #[test]
    fn test_portfolio_loss() {
        let port = PortfolioLoss::new()
            .with_obligor(LossEstimator::new(0.01, 0.40, 500_000.0))
            .with_obligor(LossEstimator::new(0.03, 0.50, 300_000.0));
        let total_el = port.total_expected_loss();
        let el1 = 0.01 * 0.40 * 500_000.0;
        let el2 = 0.03 * 0.50 * 300_000.0;
        assert!((total_el - (el1 + el2)).abs() < 1e-6);
    }

    #[test]
    fn test_herfindahl_index() {
        let port = PortfolioLoss::new()
            .with_obligor(LossEstimator::new(0.01, 0.40, 500_000.0))
            .with_obligor(LossEstimator::new(0.01, 0.40, 500_000.0));
        let hhi = port.herfindahl_index();
        assert!((hhi - 0.50).abs() < 0.01, "Two equal exposures: HHI=0.5");
    }

    #[test]
    fn test_migration_standard() {
        let mig = CreditMigration::standard_7state();
        assert_eq!(mig.rating_labels.len(), 7);
        // Default state is absorbing
        assert!((mig.migration_probability(6, 6) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_migration_row_sums() {
        let mig = CreditMigration::standard_7state();
        for (i, row) in mig.transition_matrix.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Row {i} sum should be ~1.0: {sum}"
            );
        }
    }

    #[test]
    fn test_multi_period_transition() {
        let mig = CreditMigration::standard_7state();
        let two_year = mig.multi_period(2);
        // Default still absorbing
        assert!((two_year[6][6] - 1.0).abs() < 1e-10);
        // Higher PD after 2 years
        let pd_1y = mig.default_probability(4);
        let pd_2y = two_year[4][6];
        assert!(pd_2y > pd_1y, "2-year PD should exceed 1-year PD");
    }

    #[test]
    fn test_expected_drift() {
        let mig = CreditMigration::standard_7state();
        let drift_aaa = mig.expected_drift(0);
        assert!(drift_aaa > 0.0, "AAA can only stay or downgrade: drift={drift_aaa}");
    }

    #[test]
    fn test_display_impls() {
        let cfg = CreditConfig::new();
        assert!(format!("{cfg}").contains("CreditConfig"));

        let m = MertonModel::new(100.0, 80.0, 0.25);
        assert!(format!("{m}").contains("Merton"));

        let scoring = CreditScoring::new(0.0, &[1.0]);
        assert!(format!("{scoring}").contains("CreditScoring"));
    }
}
