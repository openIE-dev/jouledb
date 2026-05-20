//! Yield curve construction and interpolation.
//!
//! Provides multiple methods for building and querying yield/discount curves:
//!
//! - [`BootstrappedCurve`] — bootstrap zero rates from deposits, futures, swaps
//! - [`NelsonSiegel`] — four-parameter parametric model (β₀, β₁, β₂, τ)
//! - [`Svensson`] — six-parameter extension of Nelson-Siegel
//! - [`CubicSplineCurve`] — natural cubic spline through observed zero rates
//! - Forward rate extraction and zero-coupon curve generation
//!
//! All arithmetic is `f64`, pure `std`-only, no external crates.

use std::fmt;

// ── Errors ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CurveError {
    InsufficientData(String),
    InvalidParameter(String),
    InterpolationFailed(String),
    NegativeMaturity,
}

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData(s) => write!(f, "insufficient data: {s}"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::InterpolationFailed(s) => write!(f, "interpolation failed: {s}"),
            Self::NegativeMaturity => write!(f, "maturity must be > 0"),
        }
    }
}

impl std::error::Error for CurveError {}

// ── Instrument types for bootstrapping ────────────────────────────

/// Market instrument used in curve bootstrapping.
#[derive(Debug, Clone)]
pub enum Instrument {
    /// Money-market deposit: maturity in years, simple rate.
    Deposit { maturity: f64, rate: f64 },
    /// Futures-implied rate: start, end in years, convexity-adjusted rate.
    Futures { start: f64, end: f64, rate: f64 },
    /// Par swap rate: maturity in years, annual coupon frequency, par rate.
    Swap { maturity: f64, frequency: usize, par_rate: f64 },
}

impl fmt::Display for Instrument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deposit { maturity, rate } => {
                write!(f, "Deposit(T={maturity:.4}, r={rate:.6})")
            }
            Self::Futures { start, end, rate } => {
                write!(f, "Futures({start:.4}→{end:.4}, r={rate:.6})")
            }
            Self::Swap { maturity, frequency, par_rate } => {
                write!(f, "Swap(T={maturity:.4}, freq={frequency}, c={par_rate:.6})")
            }
        }
    }
}

// ── Zero rate point ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct ZeroRate {
    pub maturity: f64,
    pub rate: f64,
}

impl fmt::Display for ZeroRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(T={:.4}, z={:.6})", self.maturity, self.rate)
    }
}

// ── Bootstrapped curve ────────────────────────────────────────────

/// Yield curve bootstrapped from market instruments.
#[derive(Debug, Clone)]
pub struct BootstrappedCurve {
    zeros: Vec<ZeroRate>,
}

impl BootstrappedCurve {
    /// Bootstrap a zero-coupon curve from sorted instruments.
    pub fn from_instruments(instruments: &[Instrument]) -> Result<Self, CurveError> {
        if instruments.is_empty() {
            return Err(CurveError::InsufficientData("no instruments provided".into()));
        }

        let mut zeros: Vec<ZeroRate> = Vec::new();

        for inst in instruments {
            match inst {
                Instrument::Deposit { maturity, rate } => {
                    // Simple rate → continuously compounded
                    let z = (1.0 + rate * maturity).ln() / maturity;
                    zeros.push(ZeroRate { maturity: *maturity, rate: z });
                }
                Instrument::Futures { start, end, rate } => {
                    // Forward rate segment
                    let df_start = Self::discount_from_zeros(&zeros, *start);
                    let period = end - start;
                    let df_end = df_start / (1.0 + rate * period);
                    let z = -(df_end.ln()) / end;
                    zeros.push(ZeroRate { maturity: *end, rate: z });
                }
                Instrument::Swap { maturity, frequency, par_rate } => {
                    let n = (*maturity * *frequency as f64).round() as usize;
                    let period = 1.0 / *frequency as f64;
                    let coupon = par_rate * period;

                    let mut pv_coupons = 0.0;
                    for k in 1..n {
                        let t_k = k as f64 * period;
                        let df_k = Self::discount_from_zeros(&zeros, t_k);
                        pv_coupons += coupon * df_k;
                    }

                    // Solve for discount factor at maturity
                    let df_n = (1.0 - pv_coupons) / (1.0 + coupon);
                    if df_n <= 0.0 {
                        return Err(CurveError::InvalidParameter(
                            "negative discount factor in swap bootstrap".into(),
                        ));
                    }
                    let z = -(df_n.ln()) / maturity;
                    zeros.push(ZeroRate { maturity: *maturity, rate: z });
                }
            }
        }

        zeros.sort_by(|a, b| a.maturity.partial_cmp(&b.maturity).unwrap());
        Ok(Self { zeros })
    }

    /// Interpolated zero rate at arbitrary maturity (log-linear on discount factors).
    pub fn zero_rate(&self, t: f64) -> Result<f64, CurveError> {
        if t <= 0.0 {
            return Err(CurveError::NegativeMaturity);
        }
        if self.zeros.is_empty() {
            return Err(CurveError::InsufficientData("empty curve".into()));
        }
        if self.zeros.len() == 1 {
            return Ok(self.zeros[0].rate);
        }

        // Flat extrapolation at boundaries
        if t <= self.zeros[0].maturity {
            return Ok(self.zeros[0].rate);
        }
        if t >= self.zeros.last().unwrap().maturity {
            return Ok(self.zeros.last().unwrap().rate);
        }

        // Linear interpolation on zero rates
        for i in 0..self.zeros.len() - 1 {
            let z0 = &self.zeros[i];
            let z1 = &self.zeros[i + 1];
            if t >= z0.maturity && t <= z1.maturity {
                let w = (t - z0.maturity) / (z1.maturity - z0.maturity);
                return Ok(z0.rate * (1.0 - w) + z1.rate * w);
            }
        }
        Ok(self.zeros.last().unwrap().rate)
    }

    /// Discount factor at maturity t.
    pub fn discount_factor(&self, t: f64) -> Result<f64, CurveError> {
        let z = self.zero_rate(t)?;
        Ok((-z * t).exp())
    }

    /// Forward rate between t1 and t2.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> Result<f64, CurveError> {
        if t2 <= t1 {
            return Err(CurveError::InvalidParameter("t2 must be > t1".into()));
        }
        let df1 = self.discount_factor(t1)?;
        let df2 = self.discount_factor(t2)?;
        Ok((df1 / df2).ln() / (t2 - t1))
    }

    /// Return all bootstrapped zero rates.
    pub fn zero_rates(&self) -> &[ZeroRate] {
        &self.zeros
    }

    fn discount_from_zeros(zeros: &[ZeroRate], t: f64) -> f64 {
        if zeros.is_empty() || t <= 0.0 {
            return 1.0;
        }
        // Find best matching rate
        let mut best = zeros[0].rate;
        for z in zeros {
            if z.maturity <= t {
                best = z.rate;
            }
        }
        (-best * t).exp()
    }
}

impl fmt::Display for BootstrappedCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BootstrappedCurve({} nodes)", self.zeros.len())
    }
}

// ── Nelson-Siegel model ───────────────────────────────────────────

/// Nelson-Siegel parametric yield curve: z(t) = β₀ + β₁·f₁(t) + β₂·f₂(t).
#[derive(Debug, Clone)]
pub struct NelsonSiegel {
    pub beta0: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub tau: f64,
}

impl NelsonSiegel {
    pub fn new(beta0: f64, beta1: f64, beta2: f64, tau: f64) -> Self {
        Self { beta0, beta1, beta2, tau }
    }

    pub fn with_beta0(mut self, b: f64) -> Self { self.beta0 = b; self }
    pub fn with_beta1(mut self, b: f64) -> Self { self.beta1 = b; self }
    pub fn with_beta2(mut self, b: f64) -> Self { self.beta2 = b; self }
    pub fn with_tau(mut self, t: f64) -> Self { self.tau = t; self }

    /// Zero rate at maturity t.
    pub fn zero_rate(&self, t: f64) -> Result<f64, CurveError> {
        if t <= 0.0 {
            return Err(CurveError::NegativeMaturity);
        }
        if self.tau <= 0.0 {
            return Err(CurveError::InvalidParameter("tau must be > 0".into()));
        }
        let x = t / self.tau;
        let exp_x = (-x).exp();
        let f1 = if x.abs() < 1e-10 { 1.0 } else { (1.0 - exp_x) / x };
        let f2 = f1 - exp_x;
        Ok(self.beta0 + self.beta1 * f1 + self.beta2 * f2)
    }

    /// Forward rate at maturity t.
    pub fn forward_rate(&self, t: f64) -> Result<f64, CurveError> {
        if t <= 0.0 {
            return Err(CurveError::NegativeMaturity);
        }
        let x = t / self.tau;
        let exp_x = (-x).exp();
        Ok(self.beta0 + self.beta1 * exp_x + self.beta2 * x * exp_x)
    }
}

impl fmt::Display for NelsonSiegel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "NS(β₀={:.6}, β₁={:.6}, β₂={:.6}, τ={:.4})",
            self.beta0, self.beta1, self.beta2, self.tau,
        )
    }
}

// ── Svensson model ────────────────────────────────────────────────

/// Svensson extension: adds β₃ and τ₂ to Nelson-Siegel.
#[derive(Debug, Clone)]
pub struct Svensson {
    pub beta0: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub beta3: f64,
    pub tau1: f64,
    pub tau2: f64,
}

impl Svensson {
    pub fn new(beta0: f64, beta1: f64, beta2: f64, beta3: f64, tau1: f64, tau2: f64) -> Self {
        Self { beta0, beta1, beta2, beta3, tau1, tau2 }
    }

    pub fn with_beta3(mut self, b: f64) -> Self { self.beta3 = b; self }
    pub fn with_tau2(mut self, t: f64) -> Self { self.tau2 = t; self }

    /// Zero rate at maturity t.
    pub fn zero_rate(&self, t: f64) -> Result<f64, CurveError> {
        if t <= 0.0 {
            return Err(CurveError::NegativeMaturity);
        }
        if self.tau1 <= 0.0 || self.tau2 <= 0.0 {
            return Err(CurveError::InvalidParameter("tau values must be > 0".into()));
        }
        let x1 = t / self.tau1;
        let x2 = t / self.tau2;
        let e1 = (-x1).exp();
        let e2 = (-x2).exp();
        let f1 = if x1.abs() < 1e-10 { 1.0 } else { (1.0 - e1) / x1 };
        let f2 = f1 - e1;
        let f3 = if x2.abs() < 1e-10 { 1.0 } else { (1.0 - e2) / x2 } - e2;

        Ok(self.beta0 + self.beta1 * f1 + self.beta2 * f2 + self.beta3 * f3)
    }

    /// Forward rate at maturity t.
    pub fn forward_rate(&self, t: f64) -> Result<f64, CurveError> {
        if t <= 0.0 {
            return Err(CurveError::NegativeMaturity);
        }
        let x1 = t / self.tau1;
        let x2 = t / self.tau2;
        let e1 = (-x1).exp();
        let e2 = (-x2).exp();
        Ok(self.beta0 + self.beta1 * e1 + self.beta2 * x1 * e1 + self.beta3 * x2 * e2)
    }
}

impl fmt::Display for Svensson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Svensson(β₀={:.6}, β₁={:.6}, β₂={:.6}, β₃={:.6}, τ₁={:.4}, τ₂={:.4})",
            self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2,
        )
    }
}

// ── Cubic spline curve ────────────────────────────────────────────

/// Natural cubic spline through observed zero rate points.
#[derive(Debug, Clone)]
pub struct CubicSplineCurve {
    maturities: Vec<f64>,
    rates: Vec<f64>,
    /// Second derivatives at each knot.
    m: Vec<f64>,
}

impl CubicSplineCurve {
    /// Build a natural cubic spline from (maturity, rate) pairs.
    pub fn from_points(maturities: &[f64], rates: &[f64]) -> Result<Self, CurveError> {
        let n = maturities.len();
        if n < 2 || n != rates.len() {
            return Err(CurveError::InsufficientData(
                "need ≥2 matching maturity/rate pairs".into(),
            ));
        }

        let mut t = maturities.to_vec();
        let mut r = rates.to_vec();

        // Sort by maturity
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| t[a].partial_cmp(&t[b]).unwrap());
        let t_sorted: Vec<f64> = indices.iter().map(|i| t[*i]).collect();
        let r_sorted: Vec<f64> = indices.iter().map(|i| r[*i]).collect();
        t = t_sorted;
        r = r_sorted;

        // Tridiagonal system for natural spline (m[0]=m[n-1]=0)
        let mut m = vec![0.0_f64; n];
        if n > 2 {
            let nm = n - 2;
            let mut a_diag = vec![0.0_f64; nm];
            let mut b_vec = vec![0.0_f64; nm];
            let mut c_diag = vec![0.0_f64; nm];
            let mut d_vec = vec![0.0_f64; nm];

            for i in 0..nm {
                let h0 = t[i + 1] - t[i];
                let h1 = t[i + 2] - t[i + 1];
                a_diag[i] = if i > 0 { h0 } else { 0.0 };
                b_vec[i] = 2.0 * (h0 + h1);
                c_diag[i] = if i < nm - 1 { h1 } else { 0.0 };
                d_vec[i] = 6.0 * ((r[i + 2] - r[i + 1]) / h1 - (r[i + 1] - r[i]) / h0);
            }

            // Thomas algorithm
            for i in 1..nm {
                let w = a_diag[i] / b_vec[i - 1];
                b_vec[i] -= w * c_diag[i - 1];
                d_vec[i] -= w * d_vec[i - 1];
            }
            d_vec[nm - 1] /= b_vec[nm - 1];
            for i in (0..nm - 1).rev() {
                d_vec[i] = (d_vec[i] - c_diag[i] * d_vec[i + 1]) / b_vec[i];
            }
            for i in 0..nm {
                m[i + 1] = d_vec[i];
            }
        }

        Ok(Self { maturities: t, rates: r, m })
    }

    /// Evaluate interpolated zero rate at maturity t.
    pub fn zero_rate(&self, t: f64) -> Result<f64, CurveError> {
        if t <= 0.0 {
            return Err(CurveError::NegativeMaturity);
        }
        let n = self.maturities.len();
        if t <= self.maturities[0] {
            return Ok(self.rates[0]);
        }
        if t >= self.maturities[n - 1] {
            return Ok(self.rates[n - 1]);
        }

        let mut idx = 0;
        for i in 0..n - 1 {
            if t >= self.maturities[i] && t <= self.maturities[i + 1] {
                idx = i;
                break;
            }
        }

        let h = self.maturities[idx + 1] - self.maturities[idx];
        let a = (self.maturities[idx + 1] - t) / h;
        let b = (t - self.maturities[idx]) / h;
        let rate = a * self.rates[idx]
            + b * self.rates[idx + 1]
            + ((a * a * a - a) * self.m[idx] + (b * b * b - b) * self.m[idx + 1]) * h * h / 6.0;
        Ok(rate)
    }

    pub fn knot_count(&self) -> usize {
        self.maturities.len()
    }
}

impl fmt::Display for CubicSplineCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CubicSplineCurve({} knots)", self.maturities.len())
    }
}

// ── Zero-coupon curve generation ──────────────────────────────────

/// Generate a zero-coupon curve at given maturities from a bootstrapped curve.
pub fn zero_coupon_curve(
    curve: &BootstrappedCurve,
    maturities: &[f64],
) -> Result<Vec<ZeroRate>, CurveError> {
    let mut result = Vec::with_capacity(maturities.len());
    for &t in maturities {
        let rate = curve.zero_rate(t)?;
        result.push(ZeroRate { maturity: t, rate });
    }
    Ok(result)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_bootstrap_deposits() {
        let instruments = vec![
            Instrument::Deposit { maturity: 0.25, rate: 0.04 },
            Instrument::Deposit { maturity: 0.5, rate: 0.042 },
            Instrument::Deposit { maturity: 1.0, rate: 0.045 },
        ];
        let curve = BootstrappedCurve::from_instruments(&instruments).unwrap();
        let z = curve.zero_rate(0.5).unwrap();
        assert!(z > 0.03 && z < 0.06, "zero rate = {z}");
    }

    #[test]
    fn test_discount_factor_at_zero() {
        let instruments = vec![
            Instrument::Deposit { maturity: 1.0, rate: 0.05 },
        ];
        let curve = BootstrappedCurve::from_instruments(&instruments).unwrap();
        let df = curve.discount_factor(0.001).unwrap();
        assert!(df > 0.99 && df < 1.001, "df near zero = {df}");
    }

    #[test]
    fn test_forward_rate_positive() {
        let instruments = vec![
            Instrument::Deposit { maturity: 0.5, rate: 0.04 },
            Instrument::Deposit { maturity: 1.0, rate: 0.045 },
        ];
        let curve = BootstrappedCurve::from_instruments(&instruments).unwrap();
        let fwd = curve.forward_rate(0.5, 1.0).unwrap();
        assert!(fwd > 0.0, "forward rate = {fwd}");
    }

    #[test]
    fn test_forward_rate_ordering_error() {
        let instruments = vec![
            Instrument::Deposit { maturity: 1.0, rate: 0.05 },
        ];
        let curve = BootstrappedCurve::from_instruments(&instruments).unwrap();
        assert!(curve.forward_rate(1.0, 0.5).is_err());
    }

    #[test]
    fn test_nelson_siegel_long_rate() {
        let ns = NelsonSiegel::new(0.06, -0.02, 0.01, 2.0);
        let long_rate = ns.zero_rate(30.0).unwrap();
        // At long maturities, rate → β₀
        assert!(approx(long_rate, 0.06, 0.005), "long rate ≈ β₀, got {long_rate}");
    }

    #[test]
    fn test_nelson_siegel_short_rate() {
        let ns = NelsonSiegel::new(0.06, -0.02, 0.01, 2.0);
        let short_rate = ns.zero_rate(0.01).unwrap();
        // At short maturities, rate → β₀ + β₁
        assert!(approx(short_rate, 0.04, 0.01), "short rate ≈ β₀+β₁, got {short_rate}");
    }

    #[test]
    fn test_nelson_siegel_forward() {
        let ns = NelsonSiegel::new(0.06, -0.02, 0.01, 2.0);
        let fwd = ns.forward_rate(5.0).unwrap();
        assert!(fwd > 0.0, "forward rate = {fwd}");
    }

    #[test]
    fn test_svensson_extends_ns() {
        let sv = Svensson::new(0.06, -0.02, 0.01, 0.005, 2.0, 5.0);
        let rate = sv.zero_rate(3.0).unwrap();
        assert!(rate > 0.0, "svensson rate = {rate}");
    }

    #[test]
    fn test_svensson_long_rate() {
        let sv = Svensson::new(0.06, -0.02, 0.01, 0.005, 2.0, 5.0);
        let rate = sv.zero_rate(50.0).unwrap();
        assert!(approx(rate, 0.06, 0.005), "long rate ≈ β₀, got {rate}");
    }

    #[test]
    fn test_svensson_forward() {
        let sv = Svensson::new(0.06, -0.02, 0.01, 0.005, 2.0, 5.0);
        let fwd = sv.forward_rate(5.0).unwrap();
        assert!(fwd > 0.0 && fwd < 0.10, "svensson fwd = {fwd}");
    }

    #[test]
    fn test_cubic_spline_interpolation() {
        let t = [0.5, 1.0, 2.0, 5.0, 10.0];
        let r = [0.04, 0.042, 0.045, 0.048, 0.05];
        let spline = CubicSplineCurve::from_points(&t, &r).unwrap();
        let rate = spline.zero_rate(3.0).unwrap();
        assert!(rate > 0.04 && rate < 0.05, "spline rate = {rate}");
    }

    #[test]
    fn test_cubic_spline_at_knot() {
        let t = [1.0, 2.0, 3.0, 5.0];
        let r = [0.04, 0.045, 0.048, 0.05];
        let spline = CubicSplineCurve::from_points(&t, &r).unwrap();
        let rate = spline.zero_rate(2.0).unwrap();
        assert!(approx(rate, 0.045, 0.001), "at knot: got {rate}");
    }

    #[test]
    fn test_cubic_spline_too_few_points() {
        let t = [1.0];
        let r = [0.04];
        assert!(CubicSplineCurve::from_points(&t, &r).is_err());
    }

    #[test]
    fn test_negative_maturity_error() {
        let ns = NelsonSiegel::new(0.06, -0.02, 0.01, 2.0);
        assert!(ns.zero_rate(-1.0).is_err());
    }

    #[test]
    fn test_empty_instruments_error() {
        assert!(BootstrappedCurve::from_instruments(&[]).is_err());
    }

    #[test]
    fn test_zero_coupon_curve_generation() {
        let instruments = vec![
            Instrument::Deposit { maturity: 0.5, rate: 0.04 },
            Instrument::Deposit { maturity: 1.0, rate: 0.045 },
            Instrument::Deposit { maturity: 2.0, rate: 0.05 },
        ];
        let curve = BootstrappedCurve::from_instruments(&instruments).unwrap();
        let zc = zero_coupon_curve(&curve, &[0.25, 0.5, 1.0, 1.5, 2.0]).unwrap();
        assert_eq!(zc.len(), 5);
        assert!(zc[0].rate > 0.0);
    }

    #[test]
    fn test_display_bootstrapped() {
        let instruments = vec![
            Instrument::Deposit { maturity: 1.0, rate: 0.05 },
        ];
        let curve = BootstrappedCurve::from_instruments(&instruments).unwrap();
        let s = format!("{curve}");
        assert!(s.contains("BootstrappedCurve("));
    }

    #[test]
    fn test_display_instrument() {
        let inst = Instrument::Swap { maturity: 5.0, frequency: 2, par_rate: 0.045 };
        let s = format!("{inst}");
        assert!(s.contains("Swap("));
    }

    #[test]
    fn test_bootstrap_with_swap() {
        let instruments = vec![
            Instrument::Deposit { maturity: 0.25, rate: 0.04 },
            Instrument::Deposit { maturity: 0.5, rate: 0.042 },
            Instrument::Deposit { maturity: 1.0, rate: 0.045 },
            Instrument::Swap { maturity: 2.0, frequency: 2, par_rate: 0.047 },
        ];
        let curve = BootstrappedCurve::from_instruments(&instruments).unwrap();
        let z2 = curve.zero_rate(2.0).unwrap();
        assert!(z2 > 0.03 && z2 < 0.07, "swap-bootstrapped z = {z2}");
    }
}
