//! Black-Scholes option pricing model.
//!
//! Implements the analytical Black-Scholes-Merton framework for European
//! option valuation:
//!
//! - [`BsModel`] — core pricer: European call/put, put-call parity check
//! - [`Greeks`] — first- and second-order sensitivities (delta, gamma, theta, vega, rho)
//! - [`implied_volatility`] — Newton-Raphson IV solver with configurable tolerance
//! - [`VolSurfacePoint`] / [`VolSurface`] — strike×expiry volatility grid interpolation
//!
//! All arithmetic is `f64`, pure `std`-only, no external crates.

use std::fmt;

// ── Constants ─────────────────────────────────────────────────────

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1/sqrt(2π)
const SQRT_2: f64 = std::f64::consts::SQRT_2;
const PI: f64 = std::f64::consts::PI;

// ── Errors ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BsError {
    NegativeInput(String),
    ZeroExpiry,
    ConvergenceFailure { iterations: usize, last_estimate: f64 },
    InvalidVolSurface(String),
}

impl fmt::Display for BsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeInput(s) => write!(f, "negative input: {s}"),
            Self::ZeroExpiry => write!(f, "time to expiry is zero"),
            Self::ConvergenceFailure { iterations, last_estimate } => {
                write!(f, "IV did not converge after {iterations} iters (last σ={last_estimate:.6})")
            }
            Self::InvalidVolSurface(s) => write!(f, "invalid vol surface: {s}"),
        }
    }
}

impl std::error::Error for BsError {}

// ── Normal distribution helpers ───────────────────────────────────

/// Standard normal PDF φ(x).
fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF Φ(x) via rational approximation (Abramowitz & Stegun 26.2.17).
fn norm_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    // Horner form of the Hart approximation
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-0.5 * x * x).exp();
    0.5 * (1.0 + sign * y)
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

// ── Greeks result ─────────────────────────────────────────────────

/// Greeks (first- and second-order sensitivities) for a vanilla European option.
#[derive(Debug, Clone, Copy)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

impl fmt::Display for Greeks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Δ={:.6} Γ={:.6} Θ={:.6} ν={:.6} ρ={:.6}",
            self.delta, self.gamma, self.theta, self.vega, self.rho,
        )
    }
}

// ── BsModel ───────────────────────────────────────────────────────

/// Black-Scholes model parameterised by a single set of market data.
#[derive(Debug, Clone)]
pub struct BsModel {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub volatility: f64,
    pub expiry: f64,
    pub dividend_yield: f64,
}

impl BsModel {
    pub fn new(spot: f64, strike: f64, rate: f64, volatility: f64, expiry: f64) -> Self {
        Self { spot, strike, rate, volatility, expiry, dividend_yield: 0.0 }
    }

    pub fn with_dividend_yield(mut self, q: f64) -> Self {
        self.dividend_yield = q;
        self
    }

    // ── d1, d2 ────────────────────────────────────────────────────

    fn d1(&self) -> f64 {
        let numerator = (self.spot / self.strike).ln()
            + (self.rate - self.dividend_yield + 0.5 * self.volatility * self.volatility)
                * self.expiry;
        numerator / (self.volatility * self.expiry.sqrt())
    }

    fn d2(&self) -> f64 {
        self.d1() - self.volatility * self.expiry.sqrt()
    }

    // ── Pricing ───────────────────────────────────────────────────

    /// Price a European option.
    pub fn price(&self, kind: OptionKind) -> Result<f64, BsError> {
        self.validate()?;
        let d1 = self.d1();
        let d2 = self.d2();
        let df = (-self.rate * self.expiry).exp();
        let qf = (-self.dividend_yield * self.expiry).exp();
        let price = match kind {
            OptionKind::Call => {
                self.spot * qf * norm_cdf(d1) - self.strike * df * norm_cdf(d2)
            }
            OptionKind::Put => {
                self.strike * df * norm_cdf(-d2) - self.spot * qf * norm_cdf(-d1)
            }
        };
        Ok(price)
    }

    /// Verify put-call parity: C − P = S·e^{-qT} − K·e^{-rT}.
    pub fn put_call_parity_residual(&self) -> Result<f64, BsError> {
        let c = self.price(OptionKind::Call)?;
        let p = self.price(OptionKind::Put)?;
        let df = (-self.rate * self.expiry).exp();
        let qf = (-self.dividend_yield * self.expiry).exp();
        Ok(c - p - self.spot * qf + self.strike * df)
    }

    // ── Greeks ────────────────────────────────────────────────────

    /// Compute analytical Greeks.
    pub fn greeks(&self, kind: OptionKind) -> Result<Greeks, BsError> {
        self.validate()?;
        let d1 = self.d1();
        let d2 = self.d2();
        let sqrt_t = self.expiry.sqrt();
        let df = (-self.rate * self.expiry).exp();
        let qf = (-self.dividend_yield * self.expiry).exp();

        let (delta, rho_sign, theta_sign) = match kind {
            OptionKind::Call => (qf * norm_cdf(d1), 1.0, -1.0),
            OptionKind::Put => (-qf * norm_cdf(-d1), -1.0, 1.0),
        };

        let gamma = qf * norm_pdf(d1) / (self.spot * self.volatility * sqrt_t);

        let common_theta =
            -(self.spot * qf * norm_pdf(d1) * self.volatility) / (2.0 * sqrt_t);
        let theta = match kind {
            OptionKind::Call => {
                common_theta
                    - self.rate * self.strike * df * norm_cdf(d2)
                    + self.dividend_yield * self.spot * qf * norm_cdf(d1)
            }
            OptionKind::Put => {
                common_theta
                    + self.rate * self.strike * df * norm_cdf(-d2)
                    - self.dividend_yield * self.spot * qf * norm_cdf(-d1)
            }
        };

        let vega = self.spot * qf * norm_pdf(d1) * sqrt_t;

        let rho = match kind {
            OptionKind::Call => self.strike * self.expiry * df * norm_cdf(d2),
            OptionKind::Put => -self.strike * self.expiry * df * norm_cdf(-d2),
        };

        Ok(Greeks { delta, gamma, theta, vega, rho })
    }

    // ── Validation ────────────────────────────────────────────────

    fn validate(&self) -> Result<(), BsError> {
        if self.spot <= 0.0 {
            return Err(BsError::NegativeInput("spot".into()));
        }
        if self.strike <= 0.0 {
            return Err(BsError::NegativeInput("strike".into()));
        }
        if self.volatility <= 0.0 {
            return Err(BsError::NegativeInput("volatility".into()));
        }
        if self.expiry <= 0.0 {
            return Err(BsError::ZeroExpiry);
        }
        Ok(())
    }
}

impl fmt::Display for BsModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BS(S={:.2}, K={:.2}, r={:.4}, σ={:.4}, T={:.4}, q={:.4})",
            self.spot, self.strike, self.rate, self.volatility, self.expiry, self.dividend_yield,
        )
    }
}

// ── Implied volatility ────────────────────────────────────────────

/// Newton-Raphson implied volatility solver.
pub fn implied_volatility(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    expiry: f64,
    kind: OptionKind,
    max_iters: usize,
    tol: f64,
) -> Result<f64, BsError> {
    let mut sigma = 0.25; // initial guess
    for i in 0..max_iters {
        let model = BsModel::new(spot, strike, rate, sigma, expiry);
        let price = model.price(kind)?;
        let greeks = model.greeks(kind)?;
        let vega = greeks.vega;
        if vega.abs() < 1e-14 {
            return Err(BsError::ConvergenceFailure { iterations: i, last_estimate: sigma });
        }
        let diff = price - market_price;
        if diff.abs() < tol {
            return Ok(sigma);
        }
        sigma -= diff / vega;
        if sigma <= 0.0 {
            sigma = 1e-6;
        }
    }
    Err(BsError::ConvergenceFailure { iterations: max_iters, last_estimate: sigma })
}

// ── Volatility surface ────────────────────────────────────────────

/// A single point on the implied vol surface (strike × expiry → vol).
#[derive(Debug, Clone, Copy)]
pub struct VolSurfacePoint {
    pub strike: f64,
    pub expiry: f64,
    pub vol: f64,
}

impl fmt::Display for VolSurfacePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(K={:.2}, T={:.4}, σ={:.4})", self.strike, self.expiry, self.vol)
    }
}

/// Flat grid vol surface with bilinear interpolation.
#[derive(Debug, Clone)]
pub struct VolSurface {
    points: Vec<VolSurfacePoint>,
}

impl VolSurface {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn with_point(mut self, strike: f64, expiry: f64, vol: f64) -> Self {
        self.points.push(VolSurfacePoint { strike, expiry, vol });
        self
    }

    pub fn add_point(&mut self, strike: f64, expiry: f64, vol: f64) {
        self.points.push(VolSurfacePoint { strike, expiry, vol });
    }

    /// Bilinear interpolation on the nearest four surrounding points.
    pub fn interpolate(&self, strike: f64, expiry: f64) -> Result<f64, BsError> {
        if self.points.len() < 4 {
            return Err(BsError::InvalidVolSurface(
                "need at least 4 points for interpolation".into(),
            ));
        }

        // Collect unique sorted strikes and expiries
        let mut strikes: Vec<f64> = self.points.iter().map(|p| p.strike).collect();
        strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        strikes.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let mut expiries: Vec<f64> = self.points.iter().map(|p| p.expiry).collect();
        expiries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        expiries.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        // Find bracketing indices
        let ki = bracket_index(&strikes, strike);
        let ti = bracket_index(&expiries, expiry);

        let k0 = strikes[ki];
        let k1 = strikes[(ki + 1).min(strikes.len() - 1)];
        let t0 = expiries[ti];
        let t1 = expiries[(ti + 1).min(expiries.len() - 1)];

        let v00 = self.lookup(k0, t0);
        let v01 = self.lookup(k0, t1);
        let v10 = self.lookup(k1, t0);
        let v11 = self.lookup(k1, t1);

        let dk = if (k1 - k0).abs() < 1e-12 { 0.5 } else { (strike - k0) / (k1 - k0) };
        let dt = if (t1 - t0).abs() < 1e-12 { 0.5 } else { (expiry - t0) / (t1 - t0) };

        let dk = dk.clamp(0.0, 1.0);
        let dt = dt.clamp(0.0, 1.0);

        let vol = v00 * (1.0 - dk) * (1.0 - dt)
            + v10 * dk * (1.0 - dt)
            + v01 * (1.0 - dk) * dt
            + v11 * dk * dt;
        Ok(vol)
    }

    fn lookup(&self, strike: f64, expiry: f64) -> f64 {
        self.points
            .iter()
            .min_by(|a, b| {
                let da = (a.strike - strike).abs() + (a.expiry - expiry).abs();
                let db = (b.strike - strike).abs() + (b.expiry - expiry).abs();
                da.partial_cmp(&db).unwrap()
            })
            .map(|p| p.vol)
            .unwrap_or(0.20)
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

impl fmt::Display for VolSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VolSurface({} points)", self.points.len())
    }
}

// ── Helpers ───────────────────────────────────────────────────────

fn bracket_index(sorted: &[f64], val: f64) -> usize {
    if sorted.len() <= 1 {
        return 0;
    }
    for i in 0..sorted.len() - 1 {
        if val <= sorted[i + 1] {
            return i;
        }
    }
    sorted.len() - 2
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_norm_pdf_at_zero() {
        assert!(approx(norm_pdf(0.0), INV_SQRT_2PI, 1e-10));
    }

    #[test]
    fn test_norm_cdf_symmetry() {
        assert!(approx(norm_cdf(0.0), 0.5, 1e-6));
        assert!(approx(norm_cdf(1.0) + norm_cdf(-1.0), 1.0, 1e-6));
    }

    #[test]
    fn test_call_price_atm() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let c = m.price(OptionKind::Call).unwrap();
        assert!(c > 8.0 && c < 12.0, "ATM call ≈ 10.45, got {c}");
    }

    #[test]
    fn test_put_price_atm() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let p = m.price(OptionKind::Put).unwrap();
        assert!(p > 3.0 && p < 8.0, "ATM put ≈ 5.57, got {p}");
    }

    #[test]
    fn test_put_call_parity() {
        let m = BsModel::new(100.0, 110.0, 0.05, 0.30, 0.5);
        let residual = m.put_call_parity_residual().unwrap();
        assert!(residual.abs() < 1e-10, "parity residual = {residual}");
    }

    #[test]
    fn test_deep_itm_call() {
        let m = BsModel::new(200.0, 100.0, 0.05, 0.20, 1.0);
        let c = m.price(OptionKind::Call).unwrap();
        let intrinsic = 200.0 - 100.0 * (-0.05_f64).exp();
        assert!(c >= intrinsic * 0.99, "deep ITM call ≥ intrinsic");
    }

    #[test]
    fn test_delta_call_range() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let g = m.greeks(OptionKind::Call).unwrap();
        assert!(g.delta > 0.0 && g.delta < 1.0, "call delta in (0,1), got {}", g.delta);
    }

    #[test]
    fn test_delta_put_range() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let g = m.greeks(OptionKind::Put).unwrap();
        assert!(g.delta > -1.0 && g.delta < 0.0, "put delta in (-1,0), got {}", g.delta);
    }

    #[test]
    fn test_gamma_positive() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let g = m.greeks(OptionKind::Call).unwrap();
        assert!(g.gamma > 0.0, "gamma must be positive");
    }

    #[test]
    fn test_vega_positive() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let g = m.greeks(OptionKind::Call).unwrap();
        assert!(g.vega > 0.0, "vega must be positive");
    }

    #[test]
    fn test_theta_call_negative() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let g = m.greeks(OptionKind::Call).unwrap();
        assert!(g.theta < 0.0, "ATM call theta should be negative");
    }

    #[test]
    fn test_implied_vol_roundtrip() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.25, 1.0);
        let c = m.price(OptionKind::Call).unwrap();
        let iv = implied_volatility(c, 100.0, 100.0, 0.05, 1.0, OptionKind::Call, 100, 1e-8)
            .unwrap();
        assert!(approx(iv, 0.25, 1e-6), "IV roundtrip: got {iv}");
    }

    #[test]
    fn test_iv_put_roundtrip() {
        let m = BsModel::new(100.0, 90.0, 0.03, 0.35, 0.5);
        let p = m.price(OptionKind::Put).unwrap();
        let iv = implied_volatility(p, 100.0, 90.0, 0.03, 0.5, OptionKind::Put, 100, 1e-8)
            .unwrap();
        assert!(approx(iv, 0.35, 1e-5), "IV put roundtrip: got {iv}");
    }

    #[test]
    fn test_negative_spot_error() {
        let m = BsModel::new(-1.0, 100.0, 0.05, 0.20, 1.0);
        assert!(m.price(OptionKind::Call).is_err());
    }

    #[test]
    fn test_zero_expiry_error() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 0.0);
        assert!(m.price(OptionKind::Call).is_err());
    }

    #[test]
    fn test_display_model() {
        let m = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let s = format!("{m}");
        assert!(s.contains("BS("));
    }

    #[test]
    fn test_vol_surface_interpolation() {
        let surface = VolSurface::new()
            .with_point(90.0, 0.25, 0.22)
            .with_point(90.0, 1.0, 0.20)
            .with_point(110.0, 0.25, 0.18)
            .with_point(110.0, 1.0, 0.16);
        let vol = surface.interpolate(100.0, 0.625).unwrap();
        assert!(vol > 0.15 && vol < 0.25, "interp vol = {vol}");
    }

    #[test]
    fn test_vol_surface_too_few_points() {
        let surface = VolSurface::new().with_point(100.0, 1.0, 0.20);
        assert!(surface.interpolate(100.0, 1.0).is_err());
    }

    #[test]
    fn test_dividend_yield_lowers_call() {
        let m1 = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let m2 = BsModel::new(100.0, 100.0, 0.05, 0.20, 1.0).with_dividend_yield(0.03);
        let c1 = m1.price(OptionKind::Call).unwrap();
        let c2 = m2.price(OptionKind::Call).unwrap();
        assert!(c2 < c1, "dividend yield should lower call price");
    }

    #[test]
    fn test_greeks_display() {
        let g = Greeks { delta: 0.5, gamma: 0.02, theta: -5.0, vega: 38.0, rho: 40.0 };
        let s = format!("{g}");
        assert!(s.contains("Δ="));
    }
}
