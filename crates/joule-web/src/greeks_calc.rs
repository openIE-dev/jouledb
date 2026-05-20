//! Greeks calculator — finite-difference and analytical sensitivities.
//!
//! Provides multiple approaches for computing option Greeks:
//!
//! - [`FiniteDiffGreeks`] — bumped (numerical) Greeks with configurable bump sizes
//! - [`AnalyticalGreeks`] — closed-form Greeks for European vanilla options
//! - [`PortfolioGreeks`] — aggregate Greeks across a portfolio of positions
//! - [`GreeksPnlExplain`] — decompose P&L into Greek contributions
//! - [`ScenarioGreeks`] — evaluate Greeks under stressed market parameters
//!
//! All arithmetic is `f64`, pure `std`-only, no external crates.

use std::fmt;

// ── Constants ─────────────────────────────────────────────────────

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

// ── Normal distribution helpers ───────────────────────────────────

fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn norm_cdf(x: f64) -> f64 {
    if x < -8.0 { return 0.0; }
    if x > 8.0 { return 1.0; }
    // Abramowitz & Stegun 26.2.17 rational approximation for normal CDF
    let b1 = 0.319_381_530;
    let b2 = -0.356_563_782;
    let b3 = 1.781_477_937;
    let b4 = -1.821_255_978;
    let b5 = 1.330_274_429;
    let p = 0.231_641_9;
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let poly = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t;
    let tail = poly * (-0.5 * ax * ax).exp() * INV_SQRT_2PI;
    if x >= 0.0 { 1.0 - tail } else { tail }
}

// ── Option kind ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionKind {
    Call,
    Put,
}

impl fmt::Display for OptionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Call => write!(f, "Call"),
            Self::Put => write!(f, "Put"),
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum GreeksError {
    InvalidInput(String),
    ZeroExpiry,
    EmptyPortfolio,
}

impl fmt::Display for GreeksError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::ZeroExpiry => write!(f, "time to expiry is zero"),
            Self::EmptyPortfolio => write!(f, "portfolio is empty"),
        }
    }
}

impl std::error::Error for GreeksError {}

// ── Greeks result ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
pub struct GreeksResult {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

impl fmt::Display for GreeksResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Δ={:.6} Γ={:.6} Θ={:.6} ν={:.6} ρ={:.6} vanna={:.6} volga={:.6}",
            self.delta, self.gamma, self.theta, self.vega, self.rho, self.vanna, self.volga,
        )
    }
}

// ── Black-Scholes pricer (internal) ──────────────────────────────

fn bs_price(spot: f64, strike: f64, rate: f64, vol: f64, expiry: f64, kind: OptionKind) -> f64 {
    if spot <= 0.0 || strike <= 0.0 || vol <= 0.0 || expiry <= 0.0 {
        return 0.0;
    }
    let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * expiry) / (vol * expiry.sqrt());
    let d2 = d1 - vol * expiry.sqrt();
    let df = (-rate * expiry).exp();
    match kind {
        OptionKind::Call => spot * norm_cdf(d1) - strike * df * norm_cdf(d2),
        OptionKind::Put => strike * df * norm_cdf(-d2) - spot * norm_cdf(-d1),
    }
}

// ── Finite-difference Greeks ──────────────────────────────────────

/// Bump configuration for finite-difference Greeks.
#[derive(Debug, Clone)]
pub struct BumpConfig {
    pub spot_bump: f64,
    pub vol_bump: f64,
    pub rate_bump: f64,
    pub time_bump: f64,
}

impl BumpConfig {
    pub fn new() -> Self {
        Self {
            spot_bump: 0.01,
            vol_bump: 0.01,
            rate_bump: 0.0001,
            time_bump: 1.0 / 365.0,
        }
    }

    pub fn with_spot_bump(mut self, bump: f64) -> Self {
        self.spot_bump = bump;
        self
    }

    pub fn with_vol_bump(mut self, bump: f64) -> Self {
        self.vol_bump = bump;
        self
    }

    pub fn with_rate_bump(mut self, bump: f64) -> Self {
        self.rate_bump = bump;
        self
    }

    pub fn with_time_bump(mut self, bump: f64) -> Self {
        self.time_bump = bump;
        self
    }
}

impl Default for BumpConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Greeks via finite differences (central differences).
pub struct FiniteDiffGreeks;

impl FiniteDiffGreeks {
    pub fn compute(
        spot: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
        kind: OptionKind,
        bumps: &BumpConfig,
    ) -> Result<GreeksResult, GreeksError> {
        if spot <= 0.0 || strike <= 0.0 {
            return Err(GreeksError::InvalidInput("spot/strike must be > 0".into()));
        }
        if expiry <= 0.0 {
            return Err(GreeksError::ZeroExpiry);
        }

        let base = bs_price(spot, strike, rate, vol, expiry, kind);
        let ds = spot * bumps.spot_bump;
        let dv = bumps.vol_bump;
        let dr = bumps.rate_bump;
        let dt = bumps.time_bump;

        // Delta & Gamma (central diff on spot)
        let p_up = bs_price(spot + ds, strike, rate, vol, expiry, kind);
        let p_dn = bs_price(spot - ds, strike, rate, vol, expiry, kind);
        let delta = (p_up - p_dn) / (2.0 * ds);
        let gamma = (p_up - 2.0 * base + p_dn) / (ds * ds);

        // Theta (forward diff, negative time shift)
        let p_t = bs_price(spot, strike, rate, vol, (expiry - dt).max(1e-12), kind);
        let theta = (p_t - base) / dt;

        // Vega (central diff on vol)
        let p_vu = bs_price(spot, strike, rate, vol + dv, expiry, kind);
        let p_vd = bs_price(spot, strike, rate, (vol - dv).max(1e-6), expiry, kind);
        let vega = (p_vu - p_vd) / (2.0 * dv);

        // Rho (central diff on rate)
        let p_ru = bs_price(spot, strike, rate + dr, vol, expiry, kind);
        let p_rd = bs_price(spot, strike, rate - dr, vol, expiry, kind);
        let rho = (p_ru - p_rd) / (2.0 * dr);

        // Vanna: d(delta)/d(vol) — cross partial
        let d_up = (bs_price(spot + ds, strike, rate, vol + dv, expiry, kind)
            - bs_price(spot - ds, strike, rate, vol + dv, expiry, kind))
            / (2.0 * ds);
        let d_dn = (bs_price(spot + ds, strike, rate, (vol - dv).max(1e-6), expiry, kind)
            - bs_price(spot - ds, strike, rate, (vol - dv).max(1e-6), expiry, kind))
            / (2.0 * ds);
        let vanna = (d_up - d_dn) / (2.0 * dv);

        // Volga: d²(price)/d(vol)²
        let volga = (p_vu - 2.0 * base + p_vd) / (dv * dv);

        Ok(GreeksResult { delta, gamma, theta, vega, rho, vanna, volga })
    }
}

// ── Analytical Greeks ─────────────────────────────────────────────

/// Closed-form Black-Scholes Greeks (no dividends).
pub struct AnalyticalGreeks;

impl AnalyticalGreeks {
    pub fn compute(
        spot: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
        kind: OptionKind,
    ) -> Result<GreeksResult, GreeksError> {
        if spot <= 0.0 || strike <= 0.0 || vol <= 0.0 {
            return Err(GreeksError::InvalidInput("positive values required".into()));
        }
        if expiry <= 0.0 {
            return Err(GreeksError::ZeroExpiry);
        }

        let sqrt_t = expiry.sqrt();
        let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * expiry) / (vol * sqrt_t);
        let d2 = d1 - vol * sqrt_t;
        let df = (-rate * expiry).exp();

        let delta = match kind {
            OptionKind::Call => norm_cdf(d1),
            OptionKind::Put => -norm_cdf(-d1),
        };

        let gamma = norm_pdf(d1) / (spot * vol * sqrt_t);

        let theta = match kind {
            OptionKind::Call => {
                -(spot * norm_pdf(d1) * vol) / (2.0 * sqrt_t)
                    - rate * strike * df * norm_cdf(d2)
            }
            OptionKind::Put => {
                -(spot * norm_pdf(d1) * vol) / (2.0 * sqrt_t)
                    + rate * strike * df * norm_cdf(-d2)
            }
        };

        let vega = spot * norm_pdf(d1) * sqrt_t;

        let rho = match kind {
            OptionKind::Call => strike * expiry * df * norm_cdf(d2),
            OptionKind::Put => -strike * expiry * df * norm_cdf(-d2),
        };

        let vanna = -norm_pdf(d1) * d2 / vol;
        let volga = vega * d1 * d2 / vol;

        Ok(GreeksResult { delta, gamma, theta, vega, rho, vanna, volga })
    }
}

// ── Portfolio position ────────────────────────────────────────────

/// A single option position within a portfolio.
#[derive(Debug, Clone)]
pub struct Position {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub vol: f64,
    pub expiry: f64,
    pub kind: OptionKind,
    pub quantity: f64,
}

impl Position {
    pub fn new(
        spot: f64, strike: f64, rate: f64, vol: f64, expiry: f64,
        kind: OptionKind, quantity: f64,
    ) -> Self {
        Self { spot, strike, rate, vol, expiry, kind, quantity }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "{}×{} S={:.2} K={:.2} T={:.4}",
            self.quantity, self.kind, self.spot, self.strike, self.expiry,
        )
    }
}

// ── Portfolio Greeks ──────────────────────────────────────────────

/// Aggregates Greeks across a portfolio.
pub struct PortfolioGreeks;

impl PortfolioGreeks {
    pub fn aggregate(positions: &[Position]) -> Result<GreeksResult, GreeksError> {
        if positions.is_empty() {
            return Err(GreeksError::EmptyPortfolio);
        }

        let mut total = GreeksResult::default();
        for pos in positions {
            let g = AnalyticalGreeks::compute(
                pos.spot, pos.strike, pos.rate, pos.vol, pos.expiry, pos.kind,
            )?;
            total.delta += g.delta * pos.quantity;
            total.gamma += g.gamma * pos.quantity;
            total.theta += g.theta * pos.quantity;
            total.vega += g.vega * pos.quantity;
            total.rho += g.rho * pos.quantity;
            total.vanna += g.vanna * pos.quantity;
            total.volga += g.volga * pos.quantity;
        }
        Ok(total)
    }
}

// ── Greeks P&L explain ────────────────────────────────────────────

/// Decompose P&L into Greek contributions (Taylor expansion).
#[derive(Debug, Clone)]
pub struct GreeksPnlExplain {
    pub delta_pnl: f64,
    pub gamma_pnl: f64,
    pub theta_pnl: f64,
    pub vega_pnl: f64,
    pub rho_pnl: f64,
    pub unexplained: f64,
    pub total_pnl: f64,
}

impl GreeksPnlExplain {
    /// Explain P&L given market moves.
    pub fn explain(
        greeks: &GreeksResult,
        spot_move: f64,
        vol_move: f64,
        rate_move: f64,
        time_decay: f64,
        actual_pnl: f64,
    ) -> Self {
        let delta_pnl = greeks.delta * spot_move;
        let gamma_pnl = 0.5 * greeks.gamma * spot_move * spot_move;
        let theta_pnl = greeks.theta * time_decay;
        let vega_pnl = greeks.vega * vol_move;
        let rho_pnl = greeks.rho * rate_move;
        let explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl;
        Self {
            delta_pnl,
            gamma_pnl,
            theta_pnl,
            vega_pnl,
            rho_pnl,
            unexplained: actual_pnl - explained,
            total_pnl: actual_pnl,
        }
    }
}

impl fmt::Display for GreeksPnlExplain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "P&L={:.4} [Δ={:.4} Γ={:.4} Θ={:.4} ν={:.4} ρ={:.4} unexp={:.4}]",
            self.total_pnl, self.delta_pnl, self.gamma_pnl,
            self.theta_pnl, self.vega_pnl, self.rho_pnl, self.unexplained,
        )
    }
}

// ── Scenario Greeks ───────────────────────────────────────────────

/// A scenario consisting of stressed market parameters.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub name: String,
    pub spot_shift: f64,
    pub vol_shift: f64,
    pub rate_shift: f64,
}

impl Scenario {
    pub fn new(name: &str, spot_shift: f64, vol_shift: f64, rate_shift: f64) -> Self {
        Self { name: name.to_string(), spot_shift, vol_shift, rate_shift }
    }
}

impl fmt::Display for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scenario({}: ΔS={:.2}%, Δσ={:.4}, Δr={:.4})",
            self.name, self.spot_shift * 100.0, self.vol_shift, self.rate_shift)
    }
}

/// Result of a scenario analysis.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub scenario_name: String,
    pub base_price: f64,
    pub stressed_price: f64,
    pub pnl: f64,
    pub greeks: GreeksResult,
}

impl fmt::Display for ScenarioResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "{}: base={:.4} stressed={:.4} P&L={:.4}",
            self.scenario_name, self.base_price, self.stressed_price, self.pnl,
        )
    }
}

/// Evaluate Greeks under various scenarios.
pub struct ScenarioGreeks;

impl ScenarioGreeks {
    pub fn evaluate(
        spot: f64, strike: f64, rate: f64, vol: f64, expiry: f64,
        kind: OptionKind, scenarios: &[Scenario],
    ) -> Result<Vec<ScenarioResult>, GreeksError> {
        let base_price = bs_price(spot, strike, rate, vol, expiry, kind);
        let mut results = Vec::with_capacity(scenarios.len());

        for sc in scenarios {
            let s_spot = spot * (1.0 + sc.spot_shift);
            let s_vol = (vol + sc.vol_shift).max(1e-6);
            let s_rate = rate + sc.rate_shift;
            let stressed_price = bs_price(s_spot, strike, s_rate, s_vol, expiry, kind);
            let greeks = AnalyticalGreeks::compute(s_spot, strike, s_rate, s_vol, expiry, kind)?;
            results.push(ScenarioResult {
                scenario_name: sc.name.clone(),
                base_price,
                stressed_price,
                pnl: stressed_price - base_price,
                greeks,
            });
        }
        Ok(results)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_finite_diff_call_delta() {
        let g = FiniteDiffGreeks::compute(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call, &BumpConfig::new(),
        ).unwrap();
        assert!(g.delta > 0.4 && g.delta < 0.8, "call delta = {}", g.delta);
    }

    #[test]
    fn test_finite_diff_put_delta() {
        let g = FiniteDiffGreeks::compute(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Put, &BumpConfig::new(),
        ).unwrap();
        assert!(g.delta < 0.0 && g.delta > -1.0, "put delta = {}", g.delta);
    }

    #[test]
    fn test_analytical_matches_finite_diff() {
        let anal = AnalyticalGreeks::compute(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call)
            .unwrap();
        let fd = FiniteDiffGreeks::compute(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call,
            &BumpConfig::new().with_spot_bump(0.001),
        ).unwrap();
        assert!(approx(anal.delta, fd.delta, 0.01), "delta: {} vs {}", anal.delta, fd.delta);
        assert!(approx(anal.gamma, fd.gamma, 0.05), "gamma: {} vs {}", anal.gamma, fd.gamma);
    }

    #[test]
    fn test_analytical_gamma_positive() {
        let g = AnalyticalGreeks::compute(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call)
            .unwrap();
        assert!(g.gamma > 0.0);
    }

    #[test]
    fn test_analytical_vega_positive() {
        let g = AnalyticalGreeks::compute(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call)
            .unwrap();
        assert!(g.vega > 0.0);
    }

    #[test]
    fn test_portfolio_aggregation() {
        let positions = vec![
            Position::new(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call, 10.0),
            Position::new(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Put, -10.0),
        ];
        let agg = PortfolioGreeks::aggregate(&positions).unwrap();
        // Long call + short put ≈ synthetic forward, delta ≈ 10
        assert!(agg.delta > 8.0 && agg.delta < 12.0, "portfolio delta = {}", agg.delta);
    }

    #[test]
    fn test_empty_portfolio_error() {
        assert!(PortfolioGreeks::aggregate(&[]).is_err());
    }

    #[test]
    fn test_pnl_explain_sums() {
        let g = AnalyticalGreeks::compute(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call)
            .unwrap();
        let explain = GreeksPnlExplain::explain(&g, 1.0, 0.0, 0.0, 0.0, g.delta);
        assert!(approx(explain.unexplained, 0.0, 0.05), "unexplained = {}", explain.unexplained);
    }

    #[test]
    fn test_pnl_explain_display() {
        let g = GreeksResult::default();
        let explain = GreeksPnlExplain::explain(&g, 0.0, 0.0, 0.0, 0.0, 0.0);
        let s = format!("{explain}");
        assert!(s.contains("P&L="));
    }

    #[test]
    fn test_scenario_evaluation() {
        let scenarios = vec![
            Scenario::new("crash", -0.10, 0.10, -0.01),
            Scenario::new("rally", 0.10, -0.05, 0.01),
        ];
        let results = ScenarioGreeks::evaluate(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call, &scenarios,
        ).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].pnl < 0.0, "crash should lose on call");
        assert!(results[1].pnl > 0.0, "rally should gain on call");
    }

    #[test]
    fn test_scenario_display() {
        let sc = Scenario::new("test", 0.05, 0.02, 0.001);
        let s = format!("{sc}");
        assert!(s.contains("Scenario("));
    }

    #[test]
    fn test_finite_diff_vanna() {
        let g = FiniteDiffGreeks::compute(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call, &BumpConfig::new(),
        ).unwrap();
        // Vanna should be finite
        assert!(g.vanna.is_finite(), "vanna = {}", g.vanna);
    }

    #[test]
    fn test_finite_diff_volga() {
        let g = FiniteDiffGreeks::compute(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call, &BumpConfig::new(),
        ).unwrap();
        assert!(g.volga.is_finite(), "volga = {}", g.volga);
    }

    #[test]
    fn test_zero_expiry_error() {
        assert!(AnalyticalGreeks::compute(100.0, 100.0, 0.05, 0.20, 0.0, OptionKind::Call).is_err());
    }

    #[test]
    fn test_bump_config_builder() {
        let bc = BumpConfig::new()
            .with_spot_bump(0.005)
            .with_vol_bump(0.005)
            .with_rate_bump(0.0005)
            .with_time_bump(1.0 / 252.0);
        assert!(approx(bc.spot_bump, 0.005, 1e-12));
    }

    #[test]
    fn test_greeks_result_display() {
        let g = GreeksResult::default();
        let s = format!("{g}");
        assert!(s.contains("Δ="));
    }

    #[test]
    fn test_position_display() {
        let p = Position::new(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call, 5.0);
        let s = format!("{p}");
        assert!(s.contains("Call"));
    }

    #[test]
    fn test_scenario_result_display() {
        let sr = ScenarioResult {
            scenario_name: "test".into(),
            base_price: 10.0,
            stressed_price: 8.0,
            pnl: -2.0,
            greeks: GreeksResult::default(),
        };
        let s = format!("{sr}");
        assert!(s.contains("test:"));
    }

    #[test]
    fn test_put_call_delta_relationship() {
        let call = AnalyticalGreeks::compute(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Call)
            .unwrap();
        let put = AnalyticalGreeks::compute(100.0, 100.0, 0.05, 0.20, 1.0, OptionKind::Put)
            .unwrap();
        assert!(approx(call.delta - put.delta, 1.0, 0.01), "call_Δ - put_Δ ≈ 1");
    }
}
