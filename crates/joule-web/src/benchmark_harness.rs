//! Benchmark harness — iteration timing, warmup, statistical analysis, outlier
//! detection, comparison baselines, regression detection, report generation.

use std::collections::HashMap;

// ── Timing ──────────────────────────────────────────────────────────

/// A single benchmark measurement in nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measurement {
    pub nanos: f64,
}

/// Statistical summary of a benchmark run.
#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub p95: f64,
    pub p99: f64,
    pub iqr: f64,
    pub outlier_count: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;
    if hi >= sorted.len() {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Compute statistics from a set of measurements.
pub fn compute_stats(measurements: &[Measurement]) -> Stats {
    let mut vals: Vec<f64> = measurements.iter().map(|m| m.nanos).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let count = vals.len();
    let sum: f64 = vals.iter().sum();
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    let median = percentile(&vals, 50.0);
    let variance = if count > 1 {
        vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
    } else {
        0.0
    };
    let stddev = variance.sqrt();
    let min = vals.first().copied().unwrap_or(0.0);
    let max = vals.last().copied().unwrap_or(0.0);
    let p95 = percentile(&vals, 95.0);
    let p99 = percentile(&vals, 99.0);

    let q1 = percentile(&vals, 25.0);
    let q3 = percentile(&vals, 75.0);
    let iqr = q3 - q1;
    let lo_fence = q1 - 1.5 * iqr;
    let hi_fence = q3 + 1.5 * iqr;
    let outlier_count = vals.iter().filter(|v| **v < lo_fence || **v > hi_fence).count();

    Stats { count, mean, median, stddev, min, max, p95, p99, iqr, outlier_count }
}

/// Remove outliers (beyond 1.5*IQR fences) from measurements.
pub fn remove_outliers(measurements: &[Measurement]) -> Vec<Measurement> {
    let mut vals: Vec<f64> = measurements.iter().map(|m| m.nanos).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = percentile(&vals, 25.0);
    let q3 = percentile(&vals, 75.0);
    let iqr = q3 - q1;
    let lo = q1 - 1.5 * iqr;
    let hi = q3 + 1.5 * iqr;
    measurements.iter().filter(|m| m.nanos >= lo && m.nanos <= hi).copied().collect()
}

// ── Benchmark definition ────────────────────────────────────────────

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub name: String,
    pub warmup_iterations: usize,
    pub iterations: usize,
    pub outlier_removal: bool,
}

impl BenchConfig {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            warmup_iterations: 10,
            iterations: 100,
            outlier_removal: true,
        }
    }

    pub fn warmup(mut self, n: usize) -> Self { self.warmup_iterations = n; self }
    pub fn iterations(mut self, n: usize) -> Self { self.iterations = n; self }
    pub fn keep_outliers(mut self) -> Self { self.outlier_removal = false; self }
}

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub config: BenchConfig,
    pub raw_measurements: Vec<Measurement>,
    pub cleaned_measurements: Vec<Measurement>,
    pub stats: Stats,
}

/// Run a benchmark with the given function and config.
/// The function receives the iteration index and returns a measurement in nanos.
pub fn run_bench(config: BenchConfig, mut f: impl FnMut(usize) -> f64) -> BenchResult {
    // Warmup
    for i in 0..config.warmup_iterations {
        let _ = f(i);
    }

    // Collect
    let raw: Vec<Measurement> = (0..config.iterations)
        .map(|i| Measurement { nanos: f(i) })
        .collect();

    let cleaned = if config.outlier_removal {
        remove_outliers(&raw)
    } else {
        raw.clone()
    };

    let stats = compute_stats(&cleaned);

    BenchResult {
        name: config.name.clone(),
        config,
        raw_measurements: raw,
        cleaned_measurements: cleaned,
        stats,
    }
}

// ── Comparison & regression ─────────────────────────────────────────

/// Comparison between a baseline and a candidate benchmark.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub baseline_name: String,
    pub candidate_name: String,
    pub baseline_mean: f64,
    pub candidate_mean: f64,
    pub speedup: f64,
    pub regression: bool,
    pub threshold_pct: f64,
}

/// Compare two benchmark results. Regression if candidate is slower by more than threshold_pct.
pub fn compare(baseline: &BenchResult, candidate: &BenchResult, threshold_pct: f64) -> Comparison {
    let speedup = if candidate.stats.mean > 0.0 {
        baseline.stats.mean / candidate.stats.mean
    } else {
        f64::INFINITY
    };
    let pct_change = (candidate.stats.mean - baseline.stats.mean) / baseline.stats.mean * 100.0;
    Comparison {
        baseline_name: baseline.name.clone(),
        candidate_name: candidate.name.clone(),
        baseline_mean: baseline.stats.mean,
        candidate_mean: candidate.stats.mean,
        speedup,
        regression: pct_change > threshold_pct,
        threshold_pct,
    }
}

// ── Report ──────────────────────────────────────────────────────────

/// A suite of benchmarks.
#[derive(Debug, Default)]
pub struct BenchSuite {
    pub results: Vec<BenchResult>,
    pub baselines: HashMap<String, BenchResult>,
}

impl BenchSuite {
    pub fn new() -> Self { Self::default() }

    pub fn add_result(&mut self, result: BenchResult) {
        self.results.push(result);
    }

    pub fn set_baseline(&mut self, name: &str, result: BenchResult) {
        self.baselines.insert(name.to_string(), result);
    }

    /// Generate a text report.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str("Benchmark Report\n");
        out.push_str(&"=".repeat(60));
        out.push('\n');

        for r in &self.results {
            out.push_str(&format!("\n{}\n", r.name));
            out.push_str(&"-".repeat(40));
            out.push('\n');
            out.push_str(&format!("  iterations: {} (warmup: {})\n", r.config.iterations, r.config.warmup_iterations));
            out.push_str(&format!("  mean:   {:.2} ns\n", r.stats.mean));
            out.push_str(&format!("  median: {:.2} ns\n", r.stats.median));
            out.push_str(&format!("  stddev: {:.2} ns\n", r.stats.stddev));
            out.push_str(&format!("  min:    {:.2} ns\n", r.stats.min));
            out.push_str(&format!("  max:    {:.2} ns\n", r.stats.max));
            out.push_str(&format!("  p95:    {:.2} ns\n", r.stats.p95));
            out.push_str(&format!("  p99:    {:.2} ns\n", r.stats.p99));
            out.push_str(&format!("  outliers: {}\n", r.stats.outlier_count));

            if let Some(baseline) = self.baselines.get(&r.name) {
                let cmp = compare(baseline, r, 5.0);
                out.push_str(&format!("  vs baseline: {:.2}x {}\n",
                    cmp.speedup,
                    if cmp.regression { "REGRESSION" } else { "OK" }
                ));
            }
        }
        out
    }

    /// Check for regressions against baselines. Returns names of regressed benchmarks.
    pub fn check_regressions(&self, threshold_pct: f64) -> Vec<String> {
        let mut regressions = Vec::new();
        for r in &self.results {
            if let Some(baseline) = self.baselines.get(&r.name) {
                let cmp = compare(baseline, r, threshold_pct);
                if cmp.regression {
                    regressions.push(r.name.clone());
                }
            }
        }
        regressions
    }
}

/// Format nanoseconds in a human-readable way.
pub fn format_duration(nanos: f64) -> String {
    if nanos < 1_000.0 {
        format!("{:.2} ns", nanos)
    } else if nanos < 1_000_000.0 {
        format!("{:.2} µs", nanos / 1_000.0)
    } else if nanos < 1_000_000_000.0 {
        format!("{:.2} ms", nanos / 1_000_000.0)
    } else {
        format!("{:.2} s", nanos / 1_000_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn stats_basic() {
        let m: Vec<Measurement> = (1..=10).map(|i| Measurement { nanos: i as f64 }).collect();
        let s = compute_stats(&m);
        assert_eq!(s.count, 10);
        assert!(approx_eq(s.mean, 5.5, 0.01));
        assert!(approx_eq(s.median, 5.5, 0.01));
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 10.0);
    }

    #[test]
    fn stats_single() {
        let m = vec![Measurement { nanos: 42.0 }];
        let s = compute_stats(&m);
        assert_eq!(s.count, 1);
        assert!(approx_eq(s.mean, 42.0, 0.01));
        assert!(approx_eq(s.median, 42.0, 0.01));
        assert!(approx_eq(s.stddev, 0.0, 0.01));
    }

    #[test]
    fn stats_empty() {
        let s = compute_stats(&[]);
        assert_eq!(s.count, 0);
        assert!(approx_eq(s.mean, 0.0, 0.01));
    }

    #[test]
    fn outlier_removal() {
        let mut m: Vec<Measurement> = (1..=20).map(|i| Measurement { nanos: i as f64 }).collect();
        m.push(Measurement { nanos: 1000.0 }); // outlier
        let cleaned = remove_outliers(&m);
        assert!(!cleaned.iter().any(|x| x.nanos > 100.0));
    }

    #[test]
    fn percentile_values() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(percentile(&sorted, 0.0), 1.0, 0.01));
        assert!(approx_eq(percentile(&sorted, 50.0), 3.0, 0.01));
        assert!(approx_eq(percentile(&sorted, 100.0), 5.0, 0.01));
    }

    #[test]
    fn run_bench_basic() {
        let config = BenchConfig::new("test").warmup(2).iterations(20);
        let result = run_bench(config, |i| (i as f64 + 1.0) * 10.0);
        assert_eq!(result.raw_measurements.len(), 20);
        assert!(result.stats.mean > 0.0);
    }

    #[test]
    fn run_bench_no_outlier_removal() {
        let config = BenchConfig::new("no_clean").warmup(0).iterations(10).keep_outliers();
        let result = run_bench(config, |i| if i == 5 { 99999.0 } else { 100.0 });
        assert_eq!(result.cleaned_measurements.len(), 10);
    }

    #[test]
    fn comparison_speedup() {
        let baseline = run_bench(BenchConfig::new("base").warmup(0).iterations(10), |_| 200.0);
        let candidate = run_bench(BenchConfig::new("base").warmup(0).iterations(10), |_| 100.0);
        let cmp = compare(&baseline, &candidate, 5.0);
        assert!(approx_eq(cmp.speedup, 2.0, 0.1));
        assert!(!cmp.regression);
    }

    #[test]
    fn comparison_regression() {
        let baseline = run_bench(BenchConfig::new("r").warmup(0).iterations(10), |_| 100.0);
        let candidate = run_bench(BenchConfig::new("r").warmup(0).iterations(10), |_| 200.0);
        let cmp = compare(&baseline, &candidate, 5.0);
        assert!(cmp.regression);
    }

    #[test]
    fn suite_report() {
        let mut suite = BenchSuite::new();
        let r = run_bench(BenchConfig::new("sort").warmup(0).iterations(10), |_| 500.0);
        suite.add_result(r);
        let report = suite.report();
        assert!(report.contains("sort"));
        assert!(report.contains("mean"));
    }

    #[test]
    fn suite_regression_check() {
        let mut suite = BenchSuite::new();
        let baseline = run_bench(BenchConfig::new("op").warmup(0).iterations(10), |_| 100.0);
        let candidate = run_bench(BenchConfig::new("op").warmup(0).iterations(10), |_| 200.0);
        suite.set_baseline("op", baseline);
        suite.add_result(candidate);
        let regs = suite.check_regressions(5.0);
        assert_eq!(regs, vec!["op"]);
    }

    #[test]
    fn suite_no_regression() {
        let mut suite = BenchSuite::new();
        let baseline = run_bench(BenchConfig::new("op").warmup(0).iterations(10), |_| 100.0);
        let candidate = run_bench(BenchConfig::new("op").warmup(0).iterations(10), |_| 99.0);
        suite.set_baseline("op", baseline);
        suite.add_result(candidate);
        let regs = suite.check_regressions(5.0);
        assert!(regs.is_empty());
    }

    #[test]
    fn format_ns() {
        assert_eq!(format_duration(500.0), "500.00 ns");
        assert_eq!(format_duration(1500.0), "1.50 µs");
        assert_eq!(format_duration(1_500_000.0), "1.50 ms");
        assert_eq!(format_duration(1_500_000_000.0), "1.50 s");
    }

    #[test]
    fn stddev_known_values() {
        let m = vec![
            Measurement { nanos: 2.0 },
            Measurement { nanos: 4.0 },
            Measurement { nanos: 4.0 },
            Measurement { nanos: 4.0 },
            Measurement { nanos: 5.0 },
            Measurement { nanos: 5.0 },
            Measurement { nanos: 7.0 },
            Measurement { nanos: 9.0 },
        ];
        let s = compute_stats(&m);
        assert!(approx_eq(s.mean, 5.0, 0.01));
        // Sample stddev of [2,4,4,4,5,5,7,9] = sqrt(32/7) ≈ 2.138
        assert!(approx_eq(s.stddev, 2.138, 0.01));
    }

    #[test]
    fn iqr_calculation() {
        let m: Vec<Measurement> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter().map(|n| Measurement { nanos: n }).collect();
        let s = compute_stats(&m);
        // Q1=2.75, Q3=6.25, IQR=3.5
        assert!(approx_eq(s.iqr, 3.5, 0.01));
    }

    #[test]
    fn bench_config_builder() {
        let c = BenchConfig::new("test").warmup(5).iterations(50).keep_outliers();
        assert_eq!(c.name, "test");
        assert_eq!(c.warmup_iterations, 5);
        assert_eq!(c.iterations, 50);
        assert!(!c.outlier_removal);
    }
}
