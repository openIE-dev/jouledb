//! Micro-benchmarking framework.
//!
//! Replaces `criterion`, `divan`, `benchmark.js`, and `tinybench` with a
//! pure-Rust benchmark runner. Supports warm-up iterations, measurement
//! iterations, statistical analysis (mean/median/stddev/percentiles),
//! outlier detection, comparison between runs, regression detection,
//! and formatted report output.

use std::fmt;
use std::time::Instant;

// ── Statistical Helpers ──────────────────────────────────────────

/// Compute mean of a slice of durations (in nanoseconds).
fn compute_mean(samples: &[u64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<u64>() as f64 / samples.len() as f64
}

/// Compute median of a sorted slice.
fn compute_median(sorted: &[u64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
    } else {
        sorted[mid] as f64
    }
}

/// Compute standard deviation.
fn compute_stddev(samples: &[u64], mean: f64) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|s| {
            let diff = *s as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    variance.sqrt()
}

/// Compute a percentile from a sorted slice.
fn compute_percentile(sorted: &[u64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;
    if lower == upper || upper >= sorted.len() {
        sorted[lower] as f64
    } else {
        sorted[lower] as f64 * (1.0 - frac) + sorted[upper] as f64 * frac
    }
}

// ── Outlier Detection ────────────────────────────────────────────

/// Detect outliers using the IQR (Interquartile Range) method.
/// Returns indices of outlier samples.
pub fn detect_outliers(sorted: &[u64], iqr_factor: f64) -> Vec<usize> {
    if sorted.len() < 4 {
        return vec![];
    }
    let q1 = compute_percentile(sorted, 25.0);
    let q3 = compute_percentile(sorted, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - iqr_factor * iqr;
    let upper = q3 + iqr_factor * iqr;

    sorted
        .iter()
        .enumerate()
        .filter(|(_, s)| (**s as f64) < lower || (**s as f64) > upper)
        .map(|(i, _)| i)
        .collect()
}

/// Remove outliers and return cleaned samples.
pub fn remove_outliers(samples: &[u64], iqr_factor: f64) -> Vec<u64> {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let outlier_indices = detect_outliers(&sorted, iqr_factor);
    sorted
        .iter()
        .enumerate()
        .filter(|(i, _)| !outlier_indices.contains(i))
        .map(|(_, v)| *v)
        .collect()
}

// ── Benchmark Config ─────────────────────────────────────────────

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Number of warm-up iterations (not measured).
    pub warmup_iterations: u64,
    /// Number of measured iterations.
    pub measurement_iterations: u64,
    /// IQR factor for outlier detection (1.5 = standard, 3.0 = aggressive).
    pub outlier_iqr_factor: f64,
    /// Whether to remove outliers before computing statistics.
    pub remove_outliers: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            outlier_iqr_factor: 1.5,
            remove_outliers: true,
        }
    }
}

impl BenchConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_warmup(mut self, iterations: u64) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    pub fn with_measurements(mut self, iterations: u64) -> Self {
        self.measurement_iterations = iterations;
        self
    }

    pub fn with_outlier_factor(mut self, factor: f64) -> Self {
        self.outlier_iqr_factor = factor;
        self
    }

    pub fn without_outlier_removal(mut self) -> Self {
        self.remove_outliers = false;
        self
    }
}

// ── Benchmark Result ─────────────────────────────────────────────

/// Results from a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Benchmark name.
    pub name: String,
    /// Number of measured iterations.
    pub iterations: u64,
    /// Total time in nanoseconds.
    pub total_ns: u64,
    /// Mean time per iteration in nanoseconds.
    pub mean_ns: f64,
    /// Median time per iteration in nanoseconds.
    pub median_ns: f64,
    /// Standard deviation in nanoseconds.
    pub stddev_ns: f64,
    /// Minimum sample in nanoseconds.
    pub min_ns: u64,
    /// Maximum sample in nanoseconds.
    pub max_ns: u64,
    /// 5th percentile.
    pub p5_ns: f64,
    /// 95th percentile.
    pub p95_ns: f64,
    /// 99th percentile.
    pub p99_ns: f64,
    /// Raw samples (sorted).
    pub samples: Vec<u64>,
    /// Number of outliers detected.
    pub outlier_count: usize,
}

impl BenchResult {
    /// Operations per second.
    pub fn ops_per_sec(&self) -> f64 {
        if self.mean_ns <= 0.0 {
            return f64::INFINITY;
        }
        1_000_000_000.0 / self.mean_ns
    }

    /// Throughput in MB/s given bytes processed per iteration.
    pub fn throughput_mbps(&self, bytes_per_iter: u64) -> f64 {
        if self.mean_ns <= 0.0 {
            return f64::INFINITY;
        }
        let secs_per_op = self.mean_ns / 1_000_000_000.0;
        let mb_per_op = bytes_per_iter as f64 / (1024.0 * 1024.0);
        mb_per_op / secs_per_op
    }

    /// Coefficient of variation (stddev / mean).
    pub fn cv(&self) -> f64 {
        if self.mean_ns <= 0.0 {
            return 0.0;
        }
        self.stddev_ns / self.mean_ns
    }

    /// Format duration in human-readable units.
    pub fn format_duration(ns: f64) -> String {
        if ns < 1_000.0 {
            format!("{ns:.2} ns")
        } else if ns < 1_000_000.0 {
            format!("{:.2} us", ns / 1_000.0)
        } else if ns < 1_000_000_000.0 {
            format!("{:.2} ms", ns / 1_000_000.0)
        } else {
            format!("{:.2} s", ns / 1_000_000_000.0)
        }
    }
}

impl fmt::Display for BenchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} +/- {} ({} iterations, {:.0} ops/sec)",
            self.name,
            Self::format_duration(self.mean_ns),
            Self::format_duration(self.stddev_ns),
            self.iterations,
            self.ops_per_sec()
        )
    }
}

// ── Benchmark Runner ─────────────────────────────────────────────

/// Run a benchmark for a given function.
pub fn bench(name: &str, config: &BenchConfig, mut func: impl FnMut()) -> BenchResult {
    // Warm-up
    for _ in 0..config.warmup_iterations {
        func();
    }

    // Measurement
    let mut raw_samples = Vec::with_capacity(config.measurement_iterations as usize);
    let overall_start = Instant::now();

    for _ in 0..config.measurement_iterations {
        let start = Instant::now();
        func();
        let elapsed = start.elapsed().as_nanos() as u64;
        raw_samples.push(elapsed);
    }

    let total_ns = overall_start.elapsed().as_nanos() as u64;

    // Sort for statistical analysis
    raw_samples.sort();

    // Outlier detection
    let outlier_count = detect_outliers(&raw_samples, config.outlier_iqr_factor).len();

    let analysis_samples = if config.remove_outliers && outlier_count > 0 {
        let mut cleaned = remove_outliers(&raw_samples, config.outlier_iqr_factor);
        if cleaned.is_empty() {
            cleaned = raw_samples.clone();
        }
        cleaned.sort();
        cleaned
    } else {
        raw_samples.clone()
    };

    let mean = compute_mean(&analysis_samples);
    let median = compute_median(&analysis_samples);
    let stddev = compute_stddev(&analysis_samples, mean);
    let min = analysis_samples.first().copied().unwrap_or(0);
    let max = analysis_samples.last().copied().unwrap_or(0);
    let p5 = compute_percentile(&analysis_samples, 5.0);
    let p95 = compute_percentile(&analysis_samples, 95.0);
    let p99 = compute_percentile(&analysis_samples, 99.0);

    BenchResult {
        name: name.to_string(),
        iterations: config.measurement_iterations,
        total_ns,
        mean_ns: mean,
        median_ns: median,
        stddev_ns: stddev,
        min_ns: min,
        max_ns: max,
        p5_ns: p5,
        p95_ns: p95,
        p99_ns: p99,
        samples: raw_samples,
        outlier_count,
    }
}

// ── Comparison ───────────────────────────────────────────────────

/// Result of comparing two benchmark runs.
#[derive(Debug, Clone)]
pub struct BenchComparison {
    pub baseline_name: String,
    pub candidate_name: String,
    /// Change in mean (positive = slower, negative = faster).
    pub mean_change_pct: f64,
    /// Change in median.
    pub median_change_pct: f64,
    /// Whether regression was detected (beyond threshold).
    pub is_regression: bool,
    /// Whether improvement was detected.
    pub is_improvement: bool,
    /// Threshold used for detection.
    pub threshold_pct: f64,
}

impl BenchComparison {
    pub fn compare(baseline: &BenchResult, candidate: &BenchResult, threshold_pct: f64) -> Self {
        let mean_change = if baseline.mean_ns > 0.0 {
            ((candidate.mean_ns - baseline.mean_ns) / baseline.mean_ns) * 100.0
        } else {
            0.0
        };

        let median_change = if baseline.median_ns > 0.0 {
            ((candidate.median_ns - baseline.median_ns) / baseline.median_ns) * 100.0
        } else {
            0.0
        };

        Self {
            baseline_name: baseline.name.clone(),
            candidate_name: candidate.name.clone(),
            mean_change_pct: mean_change,
            median_change_pct: median_change,
            is_regression: mean_change > threshold_pct,
            is_improvement: mean_change < -threshold_pct,
            threshold_pct,
        }
    }
}

impl fmt::Display for BenchComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction = if self.is_regression {
            "REGRESSION"
        } else if self.is_improvement {
            "IMPROVEMENT"
        } else {
            "no significant change"
        };

        write!(
            f,
            "{} vs {}: mean {:+.2}%, median {:+.2}% ({})",
            self.baseline_name,
            self.candidate_name,
            self.mean_change_pct,
            self.median_change_pct,
            direction
        )
    }
}

// ── Benchmark Suite ──────────────────────────────────────────────

/// A suite of benchmark results.
#[derive(Debug, Clone, Default)]
pub struct BenchSuite {
    results: Vec<BenchResult>,
}

impl BenchSuite {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a result.
    pub fn add(&mut self, result: BenchResult) {
        self.results.push(result);
    }

    /// Number of benchmarks in the suite.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get all results.
    pub fn results(&self) -> &[BenchResult] {
        &self.results
    }

    /// Find a result by name.
    pub fn find(&self, name: &str) -> Option<&BenchResult> {
        self.results.iter().find(|r| r.name == name)
    }

    /// Get the fastest benchmark (lowest mean).
    pub fn fastest(&self) -> Option<&BenchResult> {
        self.results
            .iter()
            .min_by(|a, b| a.mean_ns.partial_cmp(&b.mean_ns).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the slowest benchmark (highest mean).
    pub fn slowest(&self) -> Option<&BenchResult> {
        self.results
            .iter()
            .max_by(|a, b| a.mean_ns.partial_cmp(&b.mean_ns).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Compare all benchmarks against a baseline.
    pub fn compare_against(&self, baseline_name: &str, threshold_pct: f64) -> Vec<BenchComparison> {
        let baseline = match self.find(baseline_name) {
            Some(b) => b,
            None => return vec![],
        };
        self.results
            .iter()
            .filter(|r| r.name != baseline_name)
            .map(|r| BenchComparison::compare(baseline, r, threshold_pct))
            .collect()
    }

    /// Generate a formatted report.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str("Benchmark Report\n");
        out.push_str(&format!("{}\n", "=".repeat(70)));

        for r in &self.results {
            out.push_str(&format!(
                "{:<30} {:>12} +/- {:>12} ({} iters)\n",
                r.name,
                BenchResult::format_duration(r.mean_ns),
                BenchResult::format_duration(r.stddev_ns),
                r.iterations,
            ));
        }

        if let (Some(fastest), Some(slowest)) = (self.fastest(), self.slowest()) {
            out.push_str(&format!("{}\n", "-".repeat(70)));
            out.push_str(&format!("Fastest: {}\n", fastest.name));
            out.push_str(&format!("Slowest: {}\n", slowest.name));
        }

        out
    }
}

impl fmt::Display for BenchSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.report())
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_mean_basic() {
        assert!((compute_mean(&[10, 20, 30]) - 20.0).abs() < 0.01);
    }

    #[test]
    fn compute_mean_empty() {
        assert_eq!(compute_mean(&[]), 0.0);
    }

    #[test]
    fn compute_median_odd() {
        assert!((compute_median(&[10, 20, 30]) - 20.0).abs() < 0.01);
    }

    #[test]
    fn compute_median_even() {
        assert!((compute_median(&[10, 20, 30, 40]) - 25.0).abs() < 0.01);
    }

    #[test]
    fn compute_stddev_basic() {
        let samples = [10u64, 20, 30];
        let mean = compute_mean(&samples);
        let sd = compute_stddev(&samples, mean);
        assert!(sd > 0.0);
        assert!(sd < 20.0);
    }

    #[test]
    fn compute_stddev_uniform() {
        let samples = [42u64, 42, 42];
        let mean = compute_mean(&samples);
        assert_eq!(compute_stddev(&samples, mean), 0.0);
    }

    #[test]
    fn percentile_basic() {
        let sorted: Vec<u64> = (1..=100).collect();
        assert!((compute_percentile(&sorted, 50.0) - 50.0).abs() < 1.0);
        assert!(compute_percentile(&sorted, 5.0) < 10.0);
        assert!(compute_percentile(&sorted, 95.0) > 90.0);
    }

    #[test]
    fn outlier_detection() {
        let mut samples: Vec<u64> = (100..200).collect();
        samples.push(10000); // outlier
        samples.sort();
        let outliers = detect_outliers(&samples, 1.5);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn remove_outliers_cleans_data() {
        let mut samples: Vec<u64> = vec![100, 101, 102, 103, 104, 100000];
        samples.sort();
        let cleaned = remove_outliers(&samples, 1.5);
        assert!(cleaned.len() < samples.len());
        assert!(!cleaned.contains(&100000));
    }

    #[test]
    fn bench_config_builder() {
        let cfg = BenchConfig::new()
            .with_warmup(5)
            .with_measurements(50)
            .with_outlier_factor(3.0);
        assert_eq!(cfg.warmup_iterations, 5);
        assert_eq!(cfg.measurement_iterations, 50);
        assert!((cfg.outlier_iqr_factor - 3.0).abs() < 0.01);
    }

    #[test]
    fn bench_runs_function() {
        let mut counter = 0u64;
        let result = bench(
            "increment",
            &BenchConfig::new().with_warmup(2).with_measurements(10),
            || {
                counter += 1;
            },
        );
        assert_eq!(result.name, "increment");
        assert_eq!(result.iterations, 10);
        assert!(result.mean_ns > 0.0 || result.mean_ns == 0.0); // very fast op might be 0
        assert!(counter >= 12); // warmup(2) + measurement(10)
    }

    #[test]
    fn bench_result_ops_per_sec() {
        let r = BenchResult {
            name: "test".into(),
            iterations: 100,
            total_ns: 1_000_000_000,
            mean_ns: 10_000_000.0, // 10ms
            median_ns: 10_000_000.0,
            stddev_ns: 1_000.0,
            min_ns: 9_000_000,
            max_ns: 11_000_000,
            p5_ns: 9_500_000.0,
            p95_ns: 10_500_000.0,
            p99_ns: 10_900_000.0,
            samples: vec![],
            outlier_count: 0,
        };
        assert!((r.ops_per_sec() - 100.0).abs() < 0.01);
    }

    #[test]
    fn bench_result_display() {
        let r = BenchResult {
            name: "fast_op".into(),
            iterations: 1000,
            total_ns: 1000,
            mean_ns: 500.0,
            median_ns: 500.0,
            stddev_ns: 10.0,
            min_ns: 490,
            max_ns: 510,
            p5_ns: 491.0,
            p95_ns: 509.0,
            p99_ns: 510.0,
            samples: vec![],
            outlier_count: 0,
        };
        let s = format!("{r}");
        assert!(s.contains("fast_op"));
        assert!(s.contains("ns"));
    }

    #[test]
    fn format_duration_units() {
        assert!(BenchResult::format_duration(500.0).contains("ns"));
        assert!(BenchResult::format_duration(5_000.0).contains("us"));
        assert!(BenchResult::format_duration(5_000_000.0).contains("ms"));
        assert!(BenchResult::format_duration(5_000_000_000.0).contains("s"));
    }

    #[test]
    fn bench_comparison_regression() {
        let baseline = BenchResult {
            name: "v1".into(),
            iterations: 100,
            total_ns: 0,
            mean_ns: 100.0,
            median_ns: 100.0,
            stddev_ns: 5.0,
            min_ns: 90,
            max_ns: 110,
            p5_ns: 91.0,
            p95_ns: 109.0,
            p99_ns: 110.0,
            samples: vec![],
            outlier_count: 0,
        };
        let candidate = BenchResult {
            name: "v2".into(),
            mean_ns: 120.0,
            median_ns: 118.0,
            ..baseline.clone()
        };

        let cmp = BenchComparison::compare(&baseline, &candidate, 5.0);
        assert!(cmp.is_regression);
        assert!(!cmp.is_improvement);
        assert!(cmp.mean_change_pct > 15.0);
    }

    #[test]
    fn bench_comparison_improvement() {
        let baseline = BenchResult {
            name: "v1".into(),
            iterations: 100,
            total_ns: 0,
            mean_ns: 100.0,
            median_ns: 100.0,
            stddev_ns: 5.0,
            min_ns: 90,
            max_ns: 110,
            p5_ns: 91.0,
            p95_ns: 109.0,
            p99_ns: 110.0,
            samples: vec![],
            outlier_count: 0,
        };
        let candidate = BenchResult {
            name: "v2".into(),
            mean_ns: 70.0,
            median_ns: 68.0,
            ..baseline.clone()
        };

        let cmp = BenchComparison::compare(&baseline, &candidate, 5.0);
        assert!(!cmp.is_regression);
        assert!(cmp.is_improvement);
    }

    #[test]
    fn bench_suite_fastest_slowest() {
        let mut suite = BenchSuite::new();
        suite.add(BenchResult {
            name: "fast".into(),
            iterations: 100,
            total_ns: 0,
            mean_ns: 10.0,
            median_ns: 10.0,
            stddev_ns: 1.0,
            min_ns: 9,
            max_ns: 11,
            p5_ns: 9.0,
            p95_ns: 11.0,
            p99_ns: 11.0,
            samples: vec![],
            outlier_count: 0,
        });
        suite.add(BenchResult {
            name: "slow".into(),
            iterations: 100,
            total_ns: 0,
            mean_ns: 1000.0,
            median_ns: 1000.0,
            stddev_ns: 50.0,
            min_ns: 900,
            max_ns: 1100,
            p5_ns: 910.0,
            p95_ns: 1090.0,
            p99_ns: 1100.0,
            samples: vec![],
            outlier_count: 0,
        });

        assert_eq!(suite.fastest().unwrap().name, "fast");
        assert_eq!(suite.slowest().unwrap().name, "slow");
    }

    #[test]
    fn bench_suite_report_format() {
        let mut suite = BenchSuite::new();
        suite.add(BenchResult {
            name: "test_bench".into(),
            iterations: 100,
            total_ns: 50000,
            mean_ns: 500.0,
            median_ns: 490.0,
            stddev_ns: 20.0,
            min_ns: 400,
            max_ns: 600,
            p5_ns: 420.0,
            p95_ns: 580.0,
            p99_ns: 595.0,
            samples: vec![],
            outlier_count: 0,
        });
        let report = suite.report();
        assert!(report.contains("test_bench"));
        assert!(report.contains("Benchmark Report"));
    }

    #[test]
    fn bench_suite_compare_against() {
        let mut suite = BenchSuite::new();
        let base = BenchResult {
            name: "base".into(),
            iterations: 100,
            total_ns: 0,
            mean_ns: 100.0,
            median_ns: 100.0,
            stddev_ns: 5.0,
            min_ns: 90,
            max_ns: 110,
            p5_ns: 91.0,
            p95_ns: 109.0,
            p99_ns: 110.0,
            samples: vec![],
            outlier_count: 0,
        };
        let other = BenchResult {
            name: "optimized".into(),
            mean_ns: 80.0,
            median_ns: 78.0,
            ..base.clone()
        };
        suite.add(base);
        suite.add(other);

        let comparisons = suite.compare_against("base", 5.0);
        assert_eq!(comparisons.len(), 1);
        assert!(comparisons[0].is_improvement);
    }

    #[test]
    fn cv_calculation() {
        let r = BenchResult {
            name: "t".into(),
            iterations: 0,
            total_ns: 0,
            mean_ns: 100.0,
            median_ns: 100.0,
            stddev_ns: 10.0,
            min_ns: 0,
            max_ns: 0,
            p5_ns: 0.0,
            p95_ns: 0.0,
            p99_ns: 0.0,
            samples: vec![],
            outlier_count: 0,
        };
        assert!((r.cv() - 0.1).abs() < 0.001);
    }

    #[test]
    fn comparison_display() {
        let cmp = BenchComparison {
            baseline_name: "v1".into(),
            candidate_name: "v2".into(),
            mean_change_pct: 15.5,
            median_change_pct: 14.0,
            is_regression: true,
            is_improvement: false,
            threshold_pct: 5.0,
        };
        let s = format!("{cmp}");
        assert!(s.contains("REGRESSION"));
        assert!(s.contains("v1"));
        assert!(s.contains("v2"));
    }
}
