//! Benchmark tracking — tracking error, information ratio, active share,
//! sector attribution, Brinson attribution (allocation + selection +
//! interaction), and benchmark-relative analytics.
//!
//! Pure Rust, std-only. All floating-point values are `f64`.

use std::fmt;

// ── Tracking Error ──────────────────────────────────────────────

/// Compute tracking error (annualised standard deviation of active returns).
/// `portfolio_returns` and `benchmark_returns` must have the same length.
/// `periods_per_year` is typically 252 (daily) or 12 (monthly).
pub fn tracking_error(
    portfolio_returns: &[f64],
    benchmark_returns: &[f64],
    periods_per_year: f64,
) -> f64 {
    let active = active_returns(portfolio_returns, benchmark_returns);
    let std = sample_std(&active);
    std * periods_per_year.sqrt()
}

/// Compute active returns (portfolio minus benchmark).
pub fn active_returns(portfolio: &[f64], benchmark: &[f64]) -> Vec<f64> {
    portfolio
        .iter()
        .zip(benchmark.iter())
        .map(|(p, b)| p - b)
        .collect()
}

fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

fn sample_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

// ── Information Ratio ───────────────────────────────────────────

/// Information ratio = annualised mean active return / tracking error.
pub fn information_ratio(
    portfolio_returns: &[f64],
    benchmark_returns: &[f64],
    periods_per_year: f64,
) -> f64 {
    let active = active_returns(portfolio_returns, benchmark_returns);
    let te = tracking_error(portfolio_returns, benchmark_returns, periods_per_year);
    if te < 1e-15 {
        return 0.0;
    }
    let mean_active = sample_mean(&active) * periods_per_year;
    mean_active / te
}

// ── Active Share ────────────────────────────────────────────────

/// Active share = 0.5 * sum(|w_p_i - w_b_i|) for each holding.
/// Values range from 0 (index replica) to 1 (no overlap).
pub fn active_share(portfolio_weights: &[f64], benchmark_weights: &[f64]) -> f64 {
    let n = portfolio_weights.len().min(benchmark_weights.len());
    let mut sum = 0.0;
    for i in 0..n {
        sum += (portfolio_weights[i] - benchmark_weights[i]).abs();
    }
    // Account for assets only in portfolio or only in benchmark
    for &w in &portfolio_weights[n..] {
        sum += w.abs();
    }
    for &w in &benchmark_weights[n..] {
        sum += w.abs();
    }
    0.5 * sum
}

// ── SectorAttribution ───────────────────────────────────────────

/// Attribution result for a single sector.
#[derive(Debug, Clone)]
pub struct SectorAttribution {
    pub sector_name: String,
    pub portfolio_weight: f64,
    pub benchmark_weight: f64,
    pub portfolio_return: f64,
    pub benchmark_return: f64,
    pub allocation_effect: f64,
    pub selection_effect: f64,
    pub interaction_effect: f64,
    pub total_effect: f64,
}

impl SectorAttribution {
    pub fn active_weight(&self) -> f64 {
        self.portfolio_weight - self.benchmark_weight
    }

    pub fn active_return(&self) -> f64 {
        self.portfolio_return - self.benchmark_return
    }
}

impl fmt::Display for SectorAttribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sector({}): alloc={:+.2}bps, select={:+.2}bps, interact={:+.2}bps, total={:+.2}bps",
            self.sector_name,
            self.allocation_effect * 10_000.0,
            self.selection_effect * 10_000.0,
            self.interaction_effect * 10_000.0,
            self.total_effect * 10_000.0
        )
    }
}

// ── Brinson Attribution ─────────────────────────────────────────

/// Input for one sector in a Brinson attribution analysis.
#[derive(Debug, Clone)]
pub struct BrinsonInput {
    pub sector_name: String,
    pub portfolio_weight: f64,
    pub benchmark_weight: f64,
    pub portfolio_return: f64,
    pub benchmark_return: f64,
}

impl BrinsonInput {
    pub fn new(
        sector_name: &str,
        pw: f64,
        bw: f64,
        pr: f64,
        br: f64,
    ) -> Self {
        Self {
            sector_name: sector_name.to_string(),
            portfolio_weight: pw,
            benchmark_weight: bw,
            portfolio_return: pr,
            benchmark_return: br,
        }
    }
}

impl fmt::Display for BrinsonInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BrinsonInput({}, pw={:.2}%, bw={:.2}%)",
            self.sector_name,
            self.portfolio_weight * 100.0,
            self.benchmark_weight * 100.0
        )
    }
}

/// Full Brinson attribution result across all sectors.
#[derive(Debug, Clone)]
pub struct BrinsonResult {
    pub sectors: Vec<SectorAttribution>,
    pub total_allocation: f64,
    pub total_selection: f64,
    pub total_interaction: f64,
    pub total_active_return: f64,
}

impl BrinsonResult {
    pub fn sector_count(&self) -> usize {
        self.sectors.len()
    }
}

impl fmt::Display for BrinsonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Brinson(alloc={:+.2}bps, select={:+.2}bps, interact={:+.2}bps, total={:+.2}bps)",
            self.total_allocation * 10_000.0,
            self.total_selection * 10_000.0,
            self.total_interaction * 10_000.0,
            self.total_active_return * 10_000.0
        )
    }
}

/// Run Brinson attribution (BHB model: allocation + selection + interaction).
pub fn brinson_attribution(inputs: &[BrinsonInput]) -> BrinsonResult {
    let total_bm_return: f64 = inputs
        .iter()
        .map(|s| s.benchmark_weight * s.benchmark_return)
        .sum();

    let mut sectors = Vec::with_capacity(inputs.len());
    let mut total_alloc = 0.0;
    let mut total_select = 0.0;
    let mut total_interact = 0.0;

    for inp in inputs {
        let dw = inp.portfolio_weight - inp.benchmark_weight;
        let dr = inp.portfolio_return - inp.benchmark_return;

        let allocation = dw * (inp.benchmark_return - total_bm_return);
        let selection = inp.benchmark_weight * dr;
        let interaction = dw * dr;
        let total = allocation + selection + interaction;

        total_alloc += allocation;
        total_select += selection;
        total_interact += interaction;

        sectors.push(SectorAttribution {
            sector_name: inp.sector_name.clone(),
            portfolio_weight: inp.portfolio_weight,
            benchmark_weight: inp.benchmark_weight,
            portfolio_return: inp.portfolio_return,
            benchmark_return: inp.benchmark_return,
            allocation_effect: allocation,
            selection_effect: selection,
            interaction_effect: interaction,
            total_effect: total,
        });
    }

    BrinsonResult {
        sectors,
        total_allocation: total_alloc,
        total_selection: total_select,
        total_interaction: total_interact,
        total_active_return: total_alloc + total_select + total_interact,
    }
}

// ── Benchmark-Relative Analytics ────────────────────────────────

/// Summary statistics for benchmark-relative performance.
#[derive(Debug, Clone)]
pub struct BenchmarkRelativeStats {
    pub annualised_active_return: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,
    pub active_share: f64,
    pub hit_rate: f64,
    pub best_active: f64,
    pub worst_active: f64,
}

impl BenchmarkRelativeStats {
    /// Compute all benchmark-relative stats in one pass.
    pub fn compute(
        portfolio_returns: &[f64],
        benchmark_returns: &[f64],
        portfolio_weights: &[f64],
        benchmark_weights: &[f64],
        periods_per_year: f64,
    ) -> Self {
        let active = active_returns(portfolio_returns, benchmark_returns);
        let te = tracking_error(portfolio_returns, benchmark_returns, periods_per_year);
        let ir = information_ratio(portfolio_returns, benchmark_returns, periods_per_year);
        let ashare = active_share(portfolio_weights, benchmark_weights);
        let mean_active = sample_mean(&active) * periods_per_year;
        let hits = active.iter().filter(|&&a| a > 0.0).count();
        let hit_rate = if active.is_empty() {
            0.0
        } else {
            hits as f64 / active.len() as f64
        };
        let best = active.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = active.iter().cloned().fold(f64::INFINITY, f64::min);

        Self {
            annualised_active_return: mean_active,
            tracking_error: te,
            information_ratio: ir,
            active_share: ashare,
            hit_rate,
            best_active: if active.is_empty() { 0.0 } else { best },
            worst_active: if active.is_empty() { 0.0 } else { worst },
        }
    }
}

impl fmt::Display for BenchmarkRelativeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BenchmarkRelative(active={:+.2}%, TE={:.2}%, IR={:.2}, AS={:.1}%)",
            self.annualised_active_return * 100.0,
            self.tracking_error * 100.0,
            self.information_ratio,
            self.active_share * 100.0
        )
    }
}

// ── Rolling Tracking Error ──────────────────────────────────────

/// Compute rolling tracking error with a given window size.
pub fn rolling_tracking_error(
    portfolio_returns: &[f64],
    benchmark_returns: &[f64],
    window: usize,
    periods_per_year: f64,
) -> Vec<f64> {
    let n = portfolio_returns.len().min(benchmark_returns.len());
    if window == 0 || window > n {
        return Vec::new();
    }
    let active = active_returns(portfolio_returns, benchmark_returns);
    let mut result = Vec::with_capacity(n - window + 1);
    for start in 0..=(n - window) {
        let slice = &active[start..start + window];
        let std = sample_std(slice);
        result.push(std * periods_per_year.sqrt());
    }
    result
}

// ── Up/Down Capture ─────────────────────────────────────────────

/// Upside capture ratio: portfolio return in up-market periods / benchmark return.
pub fn upside_capture(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let mut port_sum = 0.0;
    let mut bench_sum = 0.0;
    let mut count = 0;
    for (&p, &b) in portfolio_returns.iter().zip(benchmark_returns.iter()) {
        if b > 0.0 {
            port_sum += p;
            bench_sum += b;
            count += 1;
        }
    }
    if count == 0 || bench_sum.abs() < 1e-15 {
        return 0.0;
    }
    (port_sum / count as f64) / (bench_sum / count as f64)
}

/// Downside capture ratio: portfolio return in down-market periods / benchmark return.
pub fn downside_capture(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let mut port_sum = 0.0;
    let mut bench_sum = 0.0;
    let mut count = 0;
    for (&p, &b) in portfolio_returns.iter().zip(benchmark_returns.iter()) {
        if b < 0.0 {
            port_sum += p;
            bench_sum += b;
            count += 1;
        }
    }
    if count == 0 || bench_sum.abs() < 1e-15 {
        return 0.0;
    }
    (port_sum / count as f64) / (bench_sum / count as f64)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> (Vec<f64>, Vec<f64>) {
        let port = vec![0.01, -0.005, 0.015, 0.008, -0.003, 0.012, 0.006, -0.001, 0.009, 0.004];
        let bench = vec![0.008, -0.003, 0.012, 0.005, -0.006, 0.010, 0.004, -0.002, 0.007, 0.003];
        (port, bench)
    }

    #[test]
    fn test_active_returns() {
        let (p, b) = sample_data();
        let ar = active_returns(&p, &b);
        assert_eq!(ar.len(), 10);
        assert!((ar[0] - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_error_positive() {
        let (p, b) = sample_data();
        let te = tracking_error(&p, &b, 252.0);
        assert!(te > 0.0);
    }

    #[test]
    fn test_tracking_error_zero_for_identical() {
        let p = vec![0.01, -0.005, 0.015];
        let te = tracking_error(&p, &p, 252.0);
        assert!(te < 1e-10);
    }

    #[test]
    fn test_information_ratio() {
        let (p, b) = sample_data();
        let ir = information_ratio(&p, &b, 252.0);
        assert!(ir > 0.0); // portfolio outperforms
    }

    #[test]
    fn test_active_share() {
        let pw = vec![0.40, 0.35, 0.25];
        let bw = vec![0.33, 0.34, 0.33];
        let ashare = active_share(&pw, &bw);
        assert!(ashare > 0.0);
        assert!(ashare <= 1.0);
    }

    #[test]
    fn test_active_share_identical() {
        let w = vec![0.5, 0.3, 0.2];
        let ashare = active_share(&w, &w);
        assert!(ashare < 1e-10);
    }

    #[test]
    fn test_brinson_input_display() {
        let bi = BrinsonInput::new("Tech", 0.30, 0.25, 0.12, 0.10);
        let s = format!("{bi}");
        assert!(s.contains("Tech"));
    }

    #[test]
    fn test_brinson_attribution_sums() {
        let inputs = vec![
            BrinsonInput::new("Tech", 0.30, 0.25, 0.12, 0.10),
            BrinsonInput::new("Healthcare", 0.25, 0.20, 0.08, 0.09),
            BrinsonInput::new("Finance", 0.20, 0.25, 0.06, 0.07),
            BrinsonInput::new("Energy", 0.15, 0.20, 0.04, 0.03),
            BrinsonInput::new("Other", 0.10, 0.10, 0.05, 0.05),
        ];
        let result = brinson_attribution(&inputs);
        assert_eq!(result.sector_count(), 5);
        let sum = result.total_allocation + result.total_selection + result.total_interaction;
        assert!((sum - result.total_active_return).abs() < 1e-12);
    }

    #[test]
    fn test_brinson_result_display() {
        let inputs = vec![
            BrinsonInput::new("A", 0.5, 0.5, 0.10, 0.08),
            BrinsonInput::new("B", 0.5, 0.5, 0.06, 0.06),
        ];
        let result = brinson_attribution(&inputs);
        let s = format!("{result}");
        assert!(s.contains("Brinson"));
    }

    #[test]
    fn test_sector_attribution_active() {
        let sa = SectorAttribution {
            sector_name: "Tech".to_string(),
            portfolio_weight: 0.30,
            benchmark_weight: 0.25,
            portfolio_return: 0.12,
            benchmark_return: 0.10,
            allocation_effect: 0.001,
            selection_effect: 0.005,
            interaction_effect: 0.001,
            total_effect: 0.007,
        };
        assert!((sa.active_weight() - 0.05).abs() < 1e-10);
        assert!((sa.active_return() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_sector_attribution_display() {
        let sa = SectorAttribution {
            sector_name: "Tech".to_string(),
            portfolio_weight: 0.30,
            benchmark_weight: 0.25,
            portfolio_return: 0.12,
            benchmark_return: 0.10,
            allocation_effect: 0.001,
            selection_effect: 0.005,
            interaction_effect: 0.001,
            total_effect: 0.007,
        };
        let s = format!("{sa}");
        assert!(s.contains("Tech"));
    }

    #[test]
    fn test_benchmark_relative_stats() {
        let (p, b) = sample_data();
        let pw = vec![0.4, 0.3, 0.3];
        let bw = vec![0.33, 0.34, 0.33];
        let stats = BenchmarkRelativeStats::compute(&p, &b, &pw, &bw, 252.0);
        assert!(stats.tracking_error > 0.0);
        assert!(stats.hit_rate >= 0.0 && stats.hit_rate <= 1.0);
    }

    #[test]
    fn test_benchmark_relative_display() {
        let (p, b) = sample_data();
        let stats = BenchmarkRelativeStats::compute(&p, &b, &[0.5, 0.5], &[0.5, 0.5], 252.0);
        let s = format!("{stats}");
        assert!(s.contains("BenchmarkRelative"));
    }

    #[test]
    fn test_rolling_tracking_error() {
        let (p, b) = sample_data();
        let rte = rolling_tracking_error(&p, &b, 5, 252.0);
        assert_eq!(rte.len(), 6);
        for &v in &rte {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_rolling_tracking_error_empty() {
        let rte = rolling_tracking_error(&[], &[], 5, 252.0);
        assert!(rte.is_empty());
    }

    #[test]
    fn test_upside_capture() {
        let (p, b) = sample_data();
        let uc = upside_capture(&p, &b);
        assert!(uc > 0.0);
    }

    #[test]
    fn test_downside_capture() {
        let (p, b) = sample_data();
        let dc = downside_capture(&p, &b);
        // portfolio falls less in down markets → capture < 1 ideally
        assert!(dc > 0.0);
    }

    #[test]
    fn test_upside_capture_no_up_periods() {
        let p = vec![-0.01, -0.02];
        let b = vec![-0.01, -0.03];
        let uc = upside_capture(&p, &b);
        assert!((uc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_brinson_no_active_return() {
        let inputs = vec![
            BrinsonInput::new("A", 0.5, 0.5, 0.05, 0.05),
            BrinsonInput::new("B", 0.5, 0.5, 0.03, 0.03),
        ];
        let result = brinson_attribution(&inputs);
        assert!(result.total_active_return.abs() < 1e-12);
    }
}
