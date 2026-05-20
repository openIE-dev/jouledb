//! Stress testing framework for financial portfolios.
//!
//! Provides tools to evaluate portfolio resilience under extreme conditions:
//!
//! - [`HistoricalScenario`] — replay known historical crises
//! - [`HypotheticalScenario`] — user-defined shock scenarios
//! - [`SensitivityAnalysis`] — bump-and-reprice sensitivity grids
//! - [`ReverseStressTest`] — find scenarios that breach a loss threshold
//! - [`StressConfig`] — builder for stress test configuration

use std::fmt;

// ── Configuration ───────────────────────────────────────────────

/// Builder for stress test parameters.
#[derive(Debug, Clone)]
pub struct StressConfig {
    pub portfolio_value: f64,
    pub loss_threshold: f64,
    pub bump_size_bps: f64,
    pub num_bumps: usize,
    pub symmetric_bumps: bool,
}

impl StressConfig {
    pub fn new() -> Self {
        Self {
            portfolio_value: 1_000_000.0,
            loss_threshold: 0.10,
            bump_size_bps: 25.0,
            num_bumps: 5,
            symmetric_bumps: true,
        }
    }

    pub fn with_portfolio_value(mut self, v: f64) -> Self {
        self.portfolio_value = v.max(0.0);
        self
    }

    pub fn with_loss_threshold(mut self, t: f64) -> Self {
        self.loss_threshold = t.clamp(0.001, 1.0);
        self
    }

    pub fn with_bump_size_bps(mut self, bps: f64) -> Self {
        self.bump_size_bps = bps.abs();
        self
    }

    pub fn with_num_bumps(mut self, n: usize) -> Self {
        self.num_bumps = n.max(1);
        self
    }

    pub fn with_symmetric_bumps(mut self, s: bool) -> Self {
        self.symmetric_bumps = s;
        self
    }
}

impl Default for StressConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StressConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StressConfig(portfolio={:.0}, threshold={:.1}%, bump={:.0}bps)",
            self.portfolio_value,
            self.loss_threshold * 100.0,
            self.bump_size_bps,
        )
    }
}

// ── Scenario Definitions ────────────────────────────────────────

/// A named market scenario with factor shocks.
#[derive(Debug, Clone)]
pub struct ScenarioShock {
    pub factor_name: String,
    pub shock_pct: f64,
}

impl ScenarioShock {
    pub fn new(name: &str, shock: f64) -> Self {
        Self {
            factor_name: name.to_string(),
            shock_pct: shock,
        }
    }
}

impl fmt::Display for ScenarioShock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:+.2}%", self.factor_name, self.shock_pct * 100.0)
    }
}

/// Outcome of applying a stress scenario.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub scenario_name: String,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub breaches_threshold: bool,
}

impl fmt::Display for ScenarioResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let flag = if self.breaches_threshold { " [BREACH]" } else { "" };
        write!(
            f,
            "{}: P&L={:+.2} ({:+.2}%){}",
            self.scenario_name,
            self.pnl,
            self.pnl_pct * 100.0,
            flag,
        )
    }
}

// ── Historical Scenario ─────────────────────────────────────────

/// Predefined historical crisis scenarios.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrisisType {
    BlackMonday1987,
    DotComBust2000,
    Gfc2008,
    CovidCrash2020,
    Custom,
}

impl fmt::Display for CrisisType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlackMonday1987 => write!(f, "Black Monday 1987"),
            Self::DotComBust2000 => write!(f, "Dot-Com Bust 2000"),
            Self::Gfc2008 => write!(f, "GFC 2008"),
            Self::CovidCrash2020 => write!(f, "COVID Crash 2020"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Replays a historical crisis scenario against current factor exposures.
#[derive(Debug, Clone)]
pub struct HistoricalScenario {
    pub crisis: CrisisType,
    pub name: String,
    pub shocks: Vec<ScenarioShock>,
}

impl HistoricalScenario {
    /// Create a standard historical scenario with predefined shocks.
    pub fn standard(crisis: CrisisType) -> Self {
        let (name, shocks) = match crisis {
            CrisisType::BlackMonday1987 => (
                "Black Monday 1987".to_string(),
                vec![
                    ScenarioShock::new("equity", -0.226),
                    ScenarioShock::new("volatility", 1.50),
                    ScenarioShock::new("rates", -0.005),
                ],
            ),
            CrisisType::DotComBust2000 => (
                "Dot-Com Bust 2000".to_string(),
                vec![
                    ScenarioShock::new("equity", -0.45),
                    ScenarioShock::new("tech_equity", -0.78),
                    ScenarioShock::new("rates", -0.015),
                ],
            ),
            CrisisType::Gfc2008 => (
                "GFC 2008".to_string(),
                vec![
                    ScenarioShock::new("equity", -0.54),
                    ScenarioShock::new("credit_spread", 0.06),
                    ScenarioShock::new("rates", -0.03),
                    ScenarioShock::new("volatility", 2.00),
                ],
            ),
            CrisisType::CovidCrash2020 => (
                "COVID Crash 2020".to_string(),
                vec![
                    ScenarioShock::new("equity", -0.34),
                    ScenarioShock::new("volatility", 3.00),
                    ScenarioShock::new("oil", -0.65),
                    ScenarioShock::new("credit_spread", 0.035),
                ],
            ),
            CrisisType::Custom => (
                "Custom Scenario".to_string(),
                Vec::new(),
            ),
        };
        Self { crisis, name, shocks }
    }

    /// Apply scenario to portfolio exposures. Returns P&L impact.
    pub fn apply(&self, exposures: &[(&str, f64)], portfolio_value: f64) -> ScenarioResult {
        let mut total_impact = 0.0;
        for (factor, exposure) in exposures {
            if let Some(shock) = self.shocks.iter().find(|s| s.factor_name == *factor) {
                total_impact += exposure * shock.shock_pct;
            }
        }
        let pnl = total_impact * portfolio_value;
        ScenarioResult {
            scenario_name: self.name.clone(),
            pnl,
            pnl_pct: total_impact,
            breaches_threshold: false,
        }
    }
}

impl fmt::Display for HistoricalScenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HistoricalScenario({}, {} shocks)", self.name, self.shocks.len())
    }
}

// ── Hypothetical Scenario ───────────────────────────────────────

/// User-defined hypothetical stress scenario.
#[derive(Debug, Clone)]
pub struct HypotheticalScenario {
    pub name: String,
    pub shocks: Vec<ScenarioShock>,
    pub description: String,
}

impl HypotheticalScenario {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            shocks: Vec::new(),
            description: String::new(),
        }
    }

    pub fn with_shock(mut self, factor: &str, shock_pct: f64) -> Self {
        self.shocks.push(ScenarioShock::new(factor, shock_pct));
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Apply this scenario to portfolio factor exposures.
    pub fn apply(&self, exposures: &[(&str, f64)], portfolio_value: f64, threshold: f64) -> ScenarioResult {
        let mut total_impact = 0.0;
        for (factor, exposure) in exposures {
            if let Some(shock) = self.shocks.iter().find(|s| s.factor_name == *factor) {
                total_impact += exposure * shock.shock_pct;
            }
        }
        let pnl = total_impact * portfolio_value;
        ScenarioResult {
            scenario_name: self.name.clone(),
            pnl,
            pnl_pct: total_impact,
            breaches_threshold: total_impact.abs() > threshold,
        }
    }
}

impl fmt::Display for HypotheticalScenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hypothetical({}): {} shocks", self.name, self.shocks.len())
    }
}

// ── Sensitivity Analysis ────────────────────────────────────────

/// Single sensitivity point from bump-and-reprice.
#[derive(Debug, Clone)]
pub struct SensitivityPoint {
    pub factor: String,
    pub bump_bps: f64,
    pub pnl_impact: f64,
    pub pnl_pct: f64,
}

impl fmt::Display for SensitivityPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@{:+.0}bps → P&L={:+.4}%",
            self.factor, self.bump_bps, self.pnl_pct * 100.0,
        )
    }
}

/// Bump-and-reprice sensitivity analysis for risk factor exposures.
#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    config: StressConfig,
}

impl SensitivityAnalysis {
    pub fn new(config: StressConfig) -> Self {
        Self { config }
    }

    /// Generate sensitivity grid for a single factor.
    pub fn analyze_factor(
        &self,
        factor: &str,
        exposure: f64,
        portfolio_value: f64,
    ) -> Vec<SensitivityPoint> {
        let mut points = Vec::new();
        let n = self.config.num_bumps;
        let bps = self.config.bump_size_bps;

        let bumps: Vec<f64> = if self.config.symmetric_bumps {
            (0..2 * n + 1)
                .map(|i| (i as f64 - n as f64) * bps)
                .collect()
        } else {
            (1..=n).map(|i| i as f64 * bps).collect()
        };

        for bump in bumps {
            let shock = bump / 10000.0;
            let impact_pct = exposure * shock;
            let impact_abs = impact_pct * portfolio_value;
            points.push(SensitivityPoint {
                factor: factor.to_string(),
                bump_bps: bump,
                pnl_impact: impact_abs,
                pnl_pct: impact_pct,
            });
        }
        points
    }

    /// Multi-factor sensitivity: bump each factor independently.
    pub fn analyze_all_factors(
        &self,
        factors: &[(&str, f64)],
        portfolio_value: f64,
    ) -> Vec<Vec<SensitivityPoint>> {
        factors
            .iter()
            .map(|(name, exp)| self.analyze_factor(name, *exp, portfolio_value))
            .collect()
    }

    /// Compute delta (first-order sensitivity) for a factor.
    pub fn compute_delta(&self, exposure: f64) -> f64 {
        exposure * self.config.bump_size_bps / 10000.0
    }

    /// Compute gamma (second-order) via central difference.
    /// `value_fn` returns portfolio value given a shock magnitude.
    pub fn compute_gamma(&self, value_fn: &dyn Fn(f64) -> f64, base: f64) -> f64 {
        let h = self.config.bump_size_bps / 10000.0;
        let v_up = value_fn(h);
        let v_down = value_fn(-h);
        let v_base = value_fn(0.0);
        if h.abs() < 1e-15 {
            return 0.0;
        }
        (v_up - 2.0 * v_base + v_down) / (h * h * base)
    }
}

impl fmt::Display for SensitivityAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SensitivityAnalysis(bump={:.0}bps, n={})",
            self.config.bump_size_bps, self.config.num_bumps,
        )
    }
}

// ── Reverse Stress Test ─────────────────────────────────────────

/// Finds factor shock magnitudes that produce a target loss.
#[derive(Debug, Clone)]
pub struct ReverseStressTest {
    config: StressConfig,
}

/// Result of a reverse stress test.
#[derive(Debug, Clone)]
pub struct ReverseStressResult {
    pub factor: String,
    pub required_shock_pct: f64,
    pub target_loss: f64,
    pub plausibility_score: f64,
}

impl fmt::Display for ReverseStressResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReverseStress({}): shock={:+.2}% to lose {:.0}, plausibility={:.2}",
            self.factor,
            self.required_shock_pct * 100.0,
            self.target_loss,
            self.plausibility_score,
        )
    }
}

impl ReverseStressTest {
    pub fn new(config: StressConfig) -> Self {
        Self { config }
    }

    /// Find the shock size for a single factor needed to breach the loss threshold.
    pub fn find_break_point(&self, factor: &str, exposure: f64) -> ReverseStressResult {
        let target_loss = self.config.portfolio_value * self.config.loss_threshold;
        let required_shock = if exposure.abs() < 1e-15 {
            f64::INFINITY
        } else {
            -target_loss / (exposure * self.config.portfolio_value)
        };
        // Simple plausibility: how many historical std devs is this shock?
        let plausibility = (-required_shock.abs() * 5.0).exp();
        ReverseStressResult {
            factor: factor.to_string(),
            required_shock_pct: required_shock,
            target_loss,
            plausibility_score: plausibility.clamp(0.0, 1.0),
        }
    }

    /// Run reverse stress test across multiple factors.
    pub fn analyze_all(&self, factors: &[(&str, f64)]) -> Vec<ReverseStressResult> {
        factors
            .iter()
            .map(|(name, exp)| self.find_break_point(name, *exp))
            .collect()
    }

    /// Find the most vulnerable factor (smallest shock to breach).
    pub fn most_vulnerable(&self, factors: &[(&str, f64)]) -> Option<ReverseStressResult> {
        self.analyze_all(factors)
            .into_iter()
            .filter(|r| r.required_shock_pct.is_finite())
            .min_by(|a, b| {
                a.required_shock_pct
                    .abs()
                    .partial_cmp(&b.required_shock_pct.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl fmt::Display for ReverseStressTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReverseStressTest(threshold={:.1}%)",
            self.config.loss_threshold * 100.0,
        )
    }
}

// ── Stress Test Suite ───────────────────────────────────────────

/// Comprehensive stress test suite combining multiple methodologies.
#[derive(Debug)]
pub struct StressTestSuite {
    config: StressConfig,
    results: Vec<ScenarioResult>,
}

impl StressTestSuite {
    pub fn new(config: StressConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run all standard historical scenarios.
    pub fn run_historical(&mut self, exposures: &[(&str, f64)]) {
        let crises = [
            CrisisType::BlackMonday1987,
            CrisisType::DotComBust2000,
            CrisisType::Gfc2008,
            CrisisType::CovidCrash2020,
        ];
        for crisis in &crises {
            let scenario = HistoricalScenario::standard(*crisis);
            let mut result = scenario.apply(exposures, self.config.portfolio_value);
            result.breaches_threshold = result.pnl_pct.abs() > self.config.loss_threshold;
            self.results.push(result);
        }
    }

    /// Run a custom hypothetical scenario.
    pub fn run_hypothetical(&mut self, scenario: &HypotheticalScenario, exposures: &[(&str, f64)]) {
        let result = scenario.apply(exposures, self.config.portfolio_value, self.config.loss_threshold);
        self.results.push(result);
    }

    /// Get all results.
    pub fn results(&self) -> &[ScenarioResult] {
        &self.results
    }

    /// Get scenarios that breached the loss threshold.
    pub fn breaches(&self) -> Vec<&ScenarioResult> {
        self.results.iter().filter(|r| r.breaches_threshold).collect()
    }

    /// Worst-case scenario result.
    pub fn worst_case(&self) -> Option<&ScenarioResult> {
        self.results
            .iter()
            .min_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap_or(std::cmp::Ordering::Equal))
    }
}

impl fmt::Display for StressTestSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StressTestSuite({} scenarios, {} breaches)",
            self.results.len(),
            self.breaches().len(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_exposures() -> Vec<(&'static str, f64)> {
        vec![
            ("equity", 0.60),
            ("rates", 0.20),
            ("credit_spread", -0.10),
            ("volatility", -0.05),
        ]
    }

    #[test]
    fn test_stress_config_builder() {
        let cfg = StressConfig::new()
            .with_portfolio_value(5_000_000.0)
            .with_loss_threshold(0.15)
            .with_bump_size_bps(50.0);
        assert!((cfg.portfolio_value - 5_000_000.0).abs() < 1e-10);
        assert!((cfg.loss_threshold - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_stress_config_display() {
        let cfg = StressConfig::new();
        let s = format!("{cfg}");
        assert!(s.contains("StressConfig"));
    }

    #[test]
    fn test_historical_scenario_gfc() {
        let scenario = HistoricalScenario::standard(CrisisType::Gfc2008);
        assert!(!scenario.shocks.is_empty());
        assert_eq!(scenario.crisis, CrisisType::Gfc2008);
    }

    #[test]
    fn test_historical_scenario_apply() {
        let scenario = HistoricalScenario::standard(CrisisType::Gfc2008);
        let exposures = sample_exposures();
        let result = scenario.apply(&exposures, 1_000_000.0);
        assert!(result.pnl < 0.0, "GFC should produce losses");
    }

    #[test]
    fn test_all_crisis_types() {
        let crises = [
            CrisisType::BlackMonday1987,
            CrisisType::DotComBust2000,
            CrisisType::Gfc2008,
            CrisisType::CovidCrash2020,
        ];
        for crisis in &crises {
            let scenario = HistoricalScenario::standard(*crisis);
            assert!(!scenario.shocks.is_empty(), "Should have shocks for {}", scenario.name);
        }
    }

    #[test]
    fn test_hypothetical_scenario() {
        let scenario = HypotheticalScenario::new("Rate Shock")
            .with_shock("rates", 0.03)
            .with_shock("equity", -0.10)
            .with_description("Sudden rate increase");
        assert_eq!(scenario.shocks.len(), 2);
    }

    #[test]
    fn test_hypothetical_apply() {
        let scenario = HypotheticalScenario::new("Equity Crash")
            .with_shock("equity", -0.30);
        let exposures = vec![("equity", 0.80)];
        let result = scenario.apply(&exposures, 1_000_000.0, 0.10);
        assert!(result.pnl < 0.0);
        assert!(result.breaches_threshold);
    }

    #[test]
    fn test_sensitivity_symmetric() {
        let cfg = StressConfig::new().with_num_bumps(3).with_bump_size_bps(25.0);
        let sa = SensitivityAnalysis::new(cfg);
        let points = sa.analyze_factor("equity", 0.5, 1_000_000.0);
        assert_eq!(points.len(), 7, "Should have 2*3+1=7 symmetric points");
    }

    #[test]
    fn test_sensitivity_asymmetric() {
        let cfg = StressConfig::new()
            .with_num_bumps(3)
            .with_symmetric_bumps(false);
        let sa = SensitivityAnalysis::new(cfg);
        let points = sa.analyze_factor("rates", 0.2, 1_000_000.0);
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn test_sensitivity_multi_factor() {
        let cfg = StressConfig::new().with_num_bumps(2);
        let sa = SensitivityAnalysis::new(cfg);
        let factors = vec![("equity", 0.6), ("rates", 0.2)];
        let grids = sa.analyze_all_factors(&factors, 1_000_000.0);
        assert_eq!(grids.len(), 2);
    }

    #[test]
    fn test_delta_computation() {
        let cfg = StressConfig::new().with_bump_size_bps(100.0);
        let sa = SensitivityAnalysis::new(cfg);
        let delta = sa.compute_delta(1.0);
        assert!((delta - 0.01).abs() < 1e-10, "100bps = 1%");
    }

    #[test]
    fn test_gamma_computation() {
        let cfg = StressConfig::new().with_bump_size_bps(10.0);
        let sa = SensitivityAnalysis::new(cfg);
        // Quadratic value function: v(x) = 1000 + 100x + 50x^2
        let value_fn = |x: f64| 1000.0 + 100.0 * x + 50.0 * x * x;
        let gamma = sa.compute_gamma(&value_fn, 1000.0);
        assert!(gamma > 0.0, "Positive convexity should give positive gamma");
    }

    #[test]
    fn test_reverse_stress_single() {
        let cfg = StressConfig::new()
            .with_portfolio_value(1_000_000.0)
            .with_loss_threshold(0.10);
        let rst = ReverseStressTest::new(cfg);
        let result = rst.find_break_point("equity", 0.80);
        assert!(result.required_shock_pct < 0.0, "Should be negative shock for long exposure");
    }

    #[test]
    fn test_reverse_stress_zero_exposure() {
        let cfg = StressConfig::new();
        let rst = ReverseStressTest::new(cfg);
        let result = rst.find_break_point("equity", 0.0);
        assert!(result.required_shock_pct.is_infinite());
    }

    #[test]
    fn test_reverse_stress_most_vulnerable() {
        let cfg = StressConfig::new().with_loss_threshold(0.10);
        let rst = ReverseStressTest::new(cfg);
        let factors = vec![("equity", 0.80), ("rates", 0.10), ("credit", 0.05)];
        let vuln = rst.most_vulnerable(&factors);
        assert!(vuln.is_some());
        assert_eq!(vuln.unwrap().factor, "equity", "Equity should be most vulnerable");
    }

    #[test]
    fn test_stress_test_suite_historical() {
        let cfg = StressConfig::new().with_portfolio_value(1_000_000.0);
        let mut suite = StressTestSuite::new(cfg);
        suite.run_historical(&sample_exposures());
        assert_eq!(suite.results().len(), 4, "Should have 4 crisis scenarios");
    }

    #[test]
    fn test_stress_suite_worst_case() {
        let cfg = StressConfig::new().with_portfolio_value(1_000_000.0);
        let mut suite = StressTestSuite::new(cfg);
        suite.run_historical(&sample_exposures());
        let worst = suite.worst_case();
        assert!(worst.is_some());
        assert!(worst.unwrap().pnl < 0.0);
    }

    #[test]
    fn test_stress_suite_with_hypothetical() {
        let cfg = StressConfig::new()
            .with_portfolio_value(1_000_000.0)
            .with_loss_threshold(0.05);
        let mut suite = StressTestSuite::new(cfg);
        let hyp = HypotheticalScenario::new("Custom")
            .with_shock("equity", -0.20);
        suite.run_hypothetical(&hyp, &sample_exposures());
        assert_eq!(suite.results().len(), 1);
    }

    #[test]
    fn test_scenario_shock_display() {
        let shock = ScenarioShock::new("equity", -0.15);
        let s = format!("{shock}");
        assert!(s.contains("equity"));
        assert!(s.contains("-15.00%"));
    }

    #[test]
    fn test_crisis_type_display() {
        assert_eq!(format!("{}", CrisisType::Gfc2008), "GFC 2008");
        assert_eq!(format!("{}", CrisisType::CovidCrash2020), "COVID Crash 2020");
    }
}
