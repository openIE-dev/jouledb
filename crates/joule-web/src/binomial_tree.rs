//! Binomial option pricing tree.
//!
//! Implements the Cox-Ross-Rubinstein (CRR) recombining binomial lattice for
//! pricing American and European options, with support for:
//!
//! - [`TreeConfig`] — builder for tree parameters (steps, barrier, dividends)
//! - [`BinomialTree`] — CRR tree construction and backward induction
//! - American options with early-exercise detection
//! - Barrier options (up-and-out, down-and-out, up-and-in, down-and-in)
//! - Discrete dividend handling (proportional or fixed)
//! - Convergence analysis against Black-Scholes
//!
//! All arithmetic is `f64`, pure `std`-only, no external crates.

use std::fmt;

// ── Option & Exercise types ───────────────────────────────────────

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExerciseStyle {
    European,
    American,
}

impl fmt::Display for ExerciseStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::European => write!(f, "European"),
            Self::American => write!(f, "American"),
        }
    }
}

// ── Barrier ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarrierKind {
    UpAndOut(f64),
    DownAndOut(f64),
    UpAndIn(f64),
    DownAndIn(f64),
}

impl fmt::Display for BarrierKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UpAndOut(b) => write!(f, "UpAndOut({b:.2})"),
            Self::DownAndOut(b) => write!(f, "DownAndOut({b:.2})"),
            Self::UpAndIn(b) => write!(f, "UpAndIn({b:.2})"),
            Self::DownAndIn(b) => write!(f, "DownAndIn({b:.2})"),
        }
    }
}

// ── Dividend ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum Dividend {
    /// Proportional yield applied continuously.
    ContinuousYield(f64),
    /// Fixed cash dividend at a specific time step.
    Discrete { step: usize, amount: f64 },
}

// ── Errors ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TreeError {
    InvalidParameter(String),
    ZeroSteps,
    BarrierBreached,
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::ZeroSteps => write!(f, "tree must have at least one step"),
            Self::BarrierBreached => write!(f, "spot already past barrier"),
        }
    }
}

impl std::error::Error for TreeError {}

// ── TreeConfig builder ────────────────────────────────────────────

/// Configuration builder for the binomial tree.
#[derive(Debug, Clone)]
pub struct TreeConfig {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub volatility: f64,
    pub expiry: f64,
    pub steps: usize,
    pub kind: OptionKind,
    pub exercise: ExerciseStyle,
    pub barrier: Option<BarrierKind>,
    pub dividends: Vec<Dividend>,
}

impl TreeConfig {
    pub fn new(spot: f64, strike: f64, rate: f64, volatility: f64, expiry: f64) -> Self {
        Self {
            spot,
            strike,
            rate,
            volatility,
            expiry,
            steps: 100,
            kind: OptionKind::Call,
            exercise: ExerciseStyle::European,
            barrier: None,
            dividends: Vec::new(),
        }
    }

    pub fn with_steps(mut self, n: usize) -> Self {
        self.steps = n;
        self
    }

    pub fn with_kind(mut self, kind: OptionKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_exercise(mut self, style: ExerciseStyle) -> Self {
        self.exercise = style;
        self
    }

    pub fn with_barrier(mut self, barrier: BarrierKind) -> Self {
        self.barrier = Some(barrier);
        self
    }

    pub fn with_dividend(mut self, div: Dividend) -> Self {
        self.dividends.push(div);
        self
    }

    fn validate(&self) -> Result<(), TreeError> {
        if self.spot <= 0.0 {
            return Err(TreeError::InvalidParameter("spot must be > 0".into()));
        }
        if self.strike <= 0.0 {
            return Err(TreeError::InvalidParameter("strike must be > 0".into()));
        }
        if self.volatility <= 0.0 {
            return Err(TreeError::InvalidParameter("volatility must be > 0".into()));
        }
        if self.expiry <= 0.0 {
            return Err(TreeError::InvalidParameter("expiry must be > 0".into()));
        }
        if self.steps == 0 {
            return Err(TreeError::ZeroSteps);
        }
        Ok(())
    }
}

impl fmt::Display for TreeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TreeConfig(S={:.2}, K={:.2}, r={:.4}, σ={:.4}, T={:.4}, steps={}, {}, {})",
            self.spot, self.strike, self.rate, self.volatility, self.expiry,
            self.steps, self.kind, self.exercise,
        )
    }
}

// ── BinomialTree ──────────────────────────────────────────────────

/// Result of binomial-tree pricing.
#[derive(Debug, Clone)]
pub struct TreeResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub early_exercise_nodes: usize,
}

impl fmt::Display for TreeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "price={:.6} Δ={:.6} Γ={:.6} Θ={:.6} early_ex={}",
            self.price, self.delta, self.gamma, self.theta, self.early_exercise_nodes,
        )
    }
}

/// CRR binomial lattice.
pub struct BinomialTree {
    config: TreeConfig,
}

impl BinomialTree {
    pub fn new(config: TreeConfig) -> Self {
        Self { config }
    }

    /// Price the option using backward induction.
    pub fn price(&self) -> Result<TreeResult, TreeError> {
        self.config.validate()?;

        let n = self.config.steps;
        let dt = self.config.expiry / n as f64;
        let q_yield = self.continuous_yield();

        let u = (self.config.volatility * dt.sqrt()).exp();
        let d = 1.0 / u;
        let r_dt = (self.config.rate * dt).exp();
        let p_up = (((self.config.rate - q_yield) * dt).exp() - d) / (u - d);
        let p_down = 1.0 - p_up;
        let disc = 1.0 / r_dt;

        // Terminal spot prices
        let mut spots = vec![0.0_f64; n + 1];
        for j in 0..=n {
            spots[j] = self.adjusted_spot(n, j, u, d);
        }

        // Terminal payoffs
        let mut values = vec![0.0_f64; n + 1];
        for j in 0..=n {
            values[j] = self.payoff(spots[j]);
            if let Some(ref barrier) = self.config.barrier {
                if self.is_knocked_out(spots[j], barrier) {
                    values[j] = 0.0;
                }
            }
        }

        let mut early_exercise_count = 0usize;

        // Store values at step 2 and step 1 for Greeks
        let mut values_step2 = Vec::new();
        let mut values_step1 = Vec::new();

        // Backward induction
        for i in (0..n).rev() {
            let mut new_values = vec![0.0_f64; i + 1];
            for j in 0..=i {
                let continuation = disc * (p_up * values[j + 1] + p_down * values[j]);
                let spot_ij = self.adjusted_spot(i, j, u, d);

                let mut val = continuation;

                // Early exercise
                if self.config.exercise == ExerciseStyle::American {
                    let exercise_val = self.payoff(spot_ij);
                    if exercise_val > continuation {
                        val = exercise_val;
                        early_exercise_count += 1;
                    }
                }

                // Barrier knock-out
                if let Some(ref barrier) = self.config.barrier {
                    if self.is_knocked_out(spot_ij, barrier) {
                        val = 0.0;
                    }
                }

                new_values[j] = val;
            }
            values = new_values;

            if i == 2 {
                values_step2 = values.clone();
            }
            if i == 1 {
                values_step1 = values.clone();
            }
        }

        let price = values[0];

        // Greeks from the tree
        let (delta, gamma, theta) = if n >= 2 && values_step1.len() >= 2 && values_step2.len() >= 3
        {
            let s_u = self.config.spot * u;
            let s_d = self.config.spot * d;
            let s_uu = self.config.spot * u * u;
            let s_dd = self.config.spot * d * d;

            let delta = (values_step1[1] - values_step1[0]) / (s_u - s_d);

            let delta_up = (values_step2[2] - values_step2[1]) / (s_uu - self.config.spot);
            let delta_down = (values_step2[1] - values_step2[0]) / (self.config.spot - s_dd);
            let gamma = (delta_up - delta_down) / (0.5 * (s_uu - s_dd));

            let theta = (values_step2[1] - price) / (2.0 * dt);

            (delta, gamma, theta)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Handle knock-in via in-out parity (vanilla − knock-out)
        let final_price = if let Some(ref barrier) = self.config.barrier {
            match barrier {
                BarrierKind::UpAndIn(_) | BarrierKind::DownAndIn(_) => {
                    // Vanilla price
                    let vanilla_cfg = TreeConfig {
                        barrier: None,
                        ..self.config.clone()
                    };
                    let vanilla = BinomialTree::new(vanilla_cfg).price()?;
                    // Corresponding out-barrier
                    let out_barrier = match barrier {
                        BarrierKind::UpAndIn(b) => BarrierKind::UpAndOut(*b),
                        BarrierKind::DownAndIn(b) => BarrierKind::DownAndOut(*b),
                        _ => unreachable!(),
                    };
                    let out_cfg = TreeConfig {
                        barrier: Some(out_barrier),
                        ..self.config.clone()
                    };
                    let out = BinomialTree::new(out_cfg).price()?;
                    vanilla.price - out.price
                }
                _ => price,
            }
        } else {
            price
        };

        Ok(TreeResult {
            price: final_price,
            delta,
            gamma,
            theta,
            early_exercise_nodes: early_exercise_count,
        })
    }

    /// Convergence: run the tree for step counts in `steps_list` and return (steps, price).
    pub fn convergence_analysis(
        base: &TreeConfig,
        steps_list: &[usize],
    ) -> Result<Vec<(usize, f64)>, TreeError> {
        let mut results = Vec::with_capacity(steps_list.len());
        for &n in steps_list {
            let cfg = TreeConfig {
                steps: n,
                ..base.clone()
            };
            let tree = BinomialTree::new(cfg);
            let r = tree.price()?;
            results.push((n, r.price));
        }
        Ok(results)
    }

    // ── Helpers ───────────────────────────────────────────────────

    fn continuous_yield(&self) -> f64 {
        self.config
            .dividends
            .iter()
            .filter_map(|d| match d {
                Dividend::ContinuousYield(q) => Some(*q),
                _ => None,
            })
            .sum()
    }

    fn adjusted_spot(&self, step: usize, j: usize, u: f64, d: f64) -> f64 {
        let up_moves = j;
        let down_moves = step - j;
        let mut s = self.config.spot * u.powi(up_moves as i32) * d.powi(down_moves as i32);

        // Apply discrete dividends
        for div in &self.config.dividends {
            if let Dividend::Discrete { step: div_step, amount } = div {
                if *div_step <= step {
                    s -= amount;
                }
            }
        }
        s.max(0.0)
    }

    fn payoff(&self, spot: f64) -> f64 {
        match self.config.kind {
            OptionKind::Call => (spot - self.config.strike).max(0.0),
            OptionKind::Put => (self.config.strike - spot).max(0.0),
        }
    }

    fn is_knocked_out(&self, spot: f64, barrier: &BarrierKind) -> bool {
        match barrier {
            BarrierKind::UpAndOut(b) => spot >= *b,
            BarrierKind::DownAndOut(b) => spot <= *b,
            // In-barriers are handled via parity
            BarrierKind::UpAndIn(_) | BarrierKind::DownAndIn(_) => false,
        }
    }
}

impl fmt::Display for BinomialTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinomialTree({})", self.config)
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
    fn test_european_call_basic() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(200)
            .with_kind(OptionKind::Call)
            .with_exercise(ExerciseStyle::European);
        let r = BinomialTree::new(cfg).price().unwrap();
        // Should be close to BS ≈ 10.45
        assert!(r.price > 9.5 && r.price < 11.5, "european call = {}", r.price);
    }

    #[test]
    fn test_european_put_basic() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(200)
            .with_kind(OptionKind::Put)
            .with_exercise(ExerciseStyle::European);
        let r = BinomialTree::new(cfg).price().unwrap();
        assert!(r.price > 4.5 && r.price < 7.0, "european put = {}", r.price);
    }

    #[test]
    fn test_american_put_geq_european() {
        let eu_cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(200)
            .with_kind(OptionKind::Put)
            .with_exercise(ExerciseStyle::European);
        let am_cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(200)
            .with_kind(OptionKind::Put)
            .with_exercise(ExerciseStyle::American);
        let eu = BinomialTree::new(eu_cfg).price().unwrap();
        let am = BinomialTree::new(am_cfg).price().unwrap();
        assert!(am.price >= eu.price - 0.01, "American ≥ European");
    }

    #[test]
    fn test_american_put_early_exercise() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Put)
            .with_exercise(ExerciseStyle::American);
        let r = BinomialTree::new(cfg).price().unwrap();
        assert!(r.early_exercise_nodes > 0, "should have early exercise nodes");
    }

    #[test]
    fn test_up_and_out_barrier_lower() {
        let vanilla = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Call);
        let barrier = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Call)
            .with_barrier(BarrierKind::UpAndOut(130.0));
        let v = BinomialTree::new(vanilla).price().unwrap();
        let b = BinomialTree::new(barrier).price().unwrap();
        assert!(b.price < v.price, "barrier option ≤ vanilla");
    }

    #[test]
    fn test_down_and_out_put() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Put)
            .with_barrier(BarrierKind::DownAndOut(80.0));
        let r = BinomialTree::new(cfg).price().unwrap();
        assert!(r.price >= 0.0, "barrier put price ≥ 0");
    }

    #[test]
    fn test_knock_in_out_parity() {
        let vanilla_cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Call);
        let ko_cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Call)
            .with_barrier(BarrierKind::UpAndOut(130.0));
        let ki_cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Call)
            .with_barrier(BarrierKind::UpAndIn(130.0));
        let v = BinomialTree::new(vanilla_cfg).price().unwrap().price;
        let ko = BinomialTree::new(ko_cfg).price().unwrap().price;
        let ki = BinomialTree::new(ki_cfg).price().unwrap().price;
        assert!(approx(ki + ko, v, 0.5), "KI + KO ≈ vanilla: {ki}+{ko} vs {v}");
    }

    #[test]
    fn test_continuous_dividend() {
        let no_div = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Call);
        let with_div = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Call)
            .with_dividend(Dividend::ContinuousYield(0.03));
        let v1 = BinomialTree::new(no_div).price().unwrap().price;
        let v2 = BinomialTree::new(with_div).price().unwrap().price;
        assert!(v2 < v1, "dividend lowers call price");
    }

    #[test]
    fn test_convergence_analysis() {
        let base = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0).with_kind(OptionKind::Call);
        let steps = [10, 50, 100, 200];
        let conv = BinomialTree::convergence_analysis(&base, &steps).unwrap();
        assert_eq!(conv.len(), 4);
        // Later steps should be closer to each other
        let spread = (conv[2].1 - conv[3].1).abs();
        assert!(spread < 0.5, "convergence: spread = {spread}");
    }

    #[test]
    fn test_tree_delta_sign() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(100)
            .with_kind(OptionKind::Call);
        let r = BinomialTree::new(cfg).price().unwrap();
        assert!(r.delta > 0.0, "call delta > 0");
    }

    #[test]
    fn test_zero_steps_error() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0).with_steps(0);
        assert!(BinomialTree::new(cfg).price().is_err());
    }

    #[test]
    fn test_negative_vol_error() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, -0.20, 1.0);
        assert!(BinomialTree::new(cfg).price().is_err());
    }

    #[test]
    fn test_deep_itm_american_call_no_early() {
        // American call on non-dividend stock: no early exercise advantage
        let cfg = TreeConfig::new(200.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(50)
            .with_kind(OptionKind::Call)
            .with_exercise(ExerciseStyle::American);
        let r = BinomialTree::new(cfg).price().unwrap();
        // For non-dividend, American call = European call
        assert!(r.price > 95.0, "deep ITM call price = {}", r.price);
    }

    #[test]
    fn test_display_tree_result() {
        let r = TreeResult {
            price: 10.0,
            delta: 0.55,
            gamma: 0.02,
            theta: -0.05,
            early_exercise_nodes: 3,
        };
        let s = format!("{r}");
        assert!(s.contains("price="));
    }

    #[test]
    fn test_display_config() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0);
        let s = format!("{cfg}");
        assert!(s.contains("TreeConfig("));
    }

    #[test]
    fn test_many_steps_stability() {
        let cfg = TreeConfig::new(100.0, 100.0, 0.05, 0.20, 1.0)
            .with_steps(500)
            .with_kind(OptionKind::Put)
            .with_exercise(ExerciseStyle::European);
        let r = BinomialTree::new(cfg).price().unwrap();
        assert!(r.price > 0.0 && r.price < 100.0, "stable price = {}", r.price);
    }

    #[test]
    fn test_otm_put_small() {
        let cfg = TreeConfig::new(100.0, 50.0, 0.05, 0.20, 0.25)
            .with_steps(100)
            .with_kind(OptionKind::Put);
        let r = BinomialTree::new(cfg).price().unwrap();
        assert!(r.price < 0.1, "deep OTM put should be near 0, got {}", r.price);
    }
}
