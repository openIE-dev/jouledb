//! Performance benchmarking.
//!
//! Replaces `benchmark.js` and `tinybench` with a pure-Rust benchmark runner.
//! Supports warm-up, statistical analysis, outlier removal (IQR), comparison
//! between runs, table formatting, and throughput calculation.

use std::fmt;
use std::time::Instant;

// ── BenchmarkResult ─────────────────────────────────────────────

/// Results from a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: u64,
    pub total_ns: u64,
    pub mean_ns: f64,
    pub median_ns: f64,
    pub stddev_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    /// Individual sample durations in nanoseconds.
    pub samples: Vec<u64>,
}

impl BenchmarkResult {
    /// Operations per second.
    pub fn ops_per_sec(&self) -> f64 {
        if self.mean_ns == 0.0 {
            return f64::INFINITY;
        }
        1_000_000_000.0 / self.mean_ns
    }

    /// Throughput: given bytes processed per iteration, compute MB/s.
    pub fn throughput_mbps(&self, bytes_per_iter: u64) -> f64 {
        if self.mean_ns == 0.0 {
            return f64::INFINITY;
        }
        let secs_per_op = self.mean_ns / 1_000_000_000.0;
        let mb_per_op = bytes_per_iter as f64 / (1024.0 * 1024.0);
        mb_per_op / secs_per_op
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2} ns/iter (+/- {:.2}), {} iterations, {:.0} ops/sec",
            self.name,
            self.mean_ns,
            self.stddev_ns,
            self.iterations,
            self.ops_per_sec()
        )
    }
}

// ── Statistical helpers ─────────────────────────────────────────

/// Compute mean of a u64 slice.
fn mean(samples: &[u64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<u64>() as f64 / samples.len() as f64
}

/// Compute median of a sorted u64 slice.
fn median(sorted: &[u64]) -> f64 {
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
fn stddev(samples: &[u64], avg: f64) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|s| {
            let diff = *s as f64 - avg;
            diff * diff
        })
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    variance.sqrt()
}

/// Remove outliers using the IQR method.
/// Returns a new sorted vector with outliers excluded.
pub fn remove_outliers_iqr(samples: &[u64]) -> Vec<u64> {
    if samples.len() < 4 {
        return samples.to_vec();
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    let q1_idx = sorted.len() / 4;
    let q3_idx = (3 * sorted.len()) / 4;
    let q1 = sorted[q1_idx] as f64;
    let q3 = sorted[q3_idx] as f64;
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;

    sorted
        .into_iter()
        .filter(|s| {
            let v = *s as f64;
            v >= lower && v <= upper
        })
        .collect()
}

// ── Benchmark runner ────────────────────────────────────────────

/// Configuration for a benchmark run.
pub struct BenchmarkConfig {
    pub name: String,
    pub warmup_iterations: u64,
    pub measure_iterations: u64,
    pub remove_outliers: bool,
}

impl BenchmarkConfig {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            warmup_iterations: 10,
            measure_iterations: 100,
            remove_outliers: true,
        }
    }

    pub fn warmup(mut self, n: u64) -> Self {
        self.warmup_iterations = n;
        self
    }

    pub fn iterations(mut self, n: u64) -> Self {
        self.measure_iterations = n;
        self
    }

    pub fn with_outlier_removal(mut self, enable: bool) -> Self {
        self.remove_outliers = enable;
        self
    }
}

/// Run a benchmark with the given configuration.
pub fn run_benchmark(config: &BenchmarkConfig, mut func: impl FnMut()) -> BenchmarkResult {
    // Warm up
    for _ in 0..config.warmup_iterations {
        func();
    }

    // Measure
    let mut samples = Vec::with_capacity(config.measure_iterations as usize);
    let total_start = Instant::now();

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        func();
        let elapsed = start.elapsed().as_nanos() as u64;
        samples.push(elapsed);
    }

    let total_ns = total_start.elapsed().as_nanos() as u64;

    // Optionally remove outliers
    let cleaned = if config.remove_outliers {
        remove_outliers_iqr(&samples)
    } else {
        let mut s = samples.clone();
        s.sort_unstable();
        s
    };

    let avg = mean(&cleaned);
    let mut sorted = cleaned.clone();
    sorted.sort_unstable();
    let med = median(&sorted);
    let sd = stddev(&cleaned, avg);
    let min_val = sorted.first().copied().unwrap_or(0);
    let max_val = sorted.last().copied().unwrap_or(0);

    BenchmarkResult {
        name: config.name.clone(),
        iterations: config.measure_iterations,
        total_ns,
        mean_ns: avg,
        median_ns: med,
        stddev_ns: sd,
        min_ns: min_val,
        max_ns: max_val,
        samples,
    }
}

/// Quick benchmark with default config.
pub fn bench(name: &str, iterations: u64, func: impl FnMut()) -> BenchmarkResult {
    let config = BenchmarkConfig::new(name).iterations(iterations).warmup(5);
    run_benchmark(&config, func)
}

// ── Comparison ──────────────────────────────────────────────────

/// Comparison between two benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub baseline_name: String,
    pub candidate_name: String,
    pub baseline_mean_ns: f64,
    pub candidate_mean_ns: f64,
    /// Positive means regression (candidate is slower); negative means speedup.
    pub change_percentage: f64,
    pub is_regression: bool,
}

impl fmt::Display for BenchmarkComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction = if self.is_regression {
            "regression"
        } else {
            "speedup"
        };
        write!(
            f,
            "{} vs {}: {:.2}% {} ({:.2} ns -> {:.2} ns)",
            self.baseline_name,
            self.candidate_name,
            self.change_percentage.abs(),
            direction,
            self.baseline_mean_ns,
            self.candidate_mean_ns
        )
    }
}

/// Compare two benchmark results.
pub fn compare_results(baseline: &BenchmarkResult, candidate: &BenchmarkResult) -> BenchmarkComparison {
    let change_pct = if baseline.mean_ns == 0.0 {
        0.0
    } else {
        ((candidate.mean_ns - baseline.mean_ns) / baseline.mean_ns) * 100.0
    };

    BenchmarkComparison {
        baseline_name: baseline.name.clone(),
        candidate_name: candidate.name.clone(),
        baseline_mean_ns: baseline.mean_ns,
        candidate_mean_ns: candidate.mean_ns,
        change_percentage: change_pct,
        is_regression: change_pct > 0.0,
    }
}

// ── Table formatting ────────────────────────────────────────────

/// Format a slice of benchmark results as an ASCII table.
pub fn format_table(results: &[BenchmarkResult]) -> String {
    if results.is_empty() {
        return String::from("(no benchmarks)");
    }

    let name_width = results
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(4)
        .max(4);

    let mut out = String::new();
    let header = format!(
        "{:<width$}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Name",
        "Mean (ns)",
        "Median (ns)",
        "Stddev (ns)",
        "Min (ns)",
        "Ops/sec",
        width = name_width
    );
    let sep = "-".repeat(header.len());
    out.push_str(&header);
    out.push('\n');
    out.push_str(&sep);
    out.push('\n');

    for r in results {
        let line = format!(
            "{:<width$}  {:>12.2}  {:>12.2}  {:>12.2}  {:>12}  {:>12.0}",
            r.name,
            r.mean_ns,
            r.median_ns,
            r.stddev_ns,
            r.min_ns,
            r.ops_per_sec(),
            width = name_width
        );
        out.push_str(&line);
        out.push('\n');
    }

    out
}

/// Build a BenchmarkResult from raw sample data (for testing/reconstruction).
pub fn result_from_samples(name: &str, samples: &[u64]) -> BenchmarkResult {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let avg = mean(samples);
    let med = median(&sorted);
    let sd = stddev(samples, avg);

    BenchmarkResult {
        name: name.to_string(),
        iterations: samples.len() as u64,
        total_ns: samples.iter().sum(),
        mean_ns: avg,
        median_ns: med,
        stddev_ns: sd,
        min_ns: sorted.first().copied().unwrap_or(0),
        max_ns: sorted.last().copied().unwrap_or(0),
        samples: samples.to_vec(),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_calculation() {
        assert!((mean(&[10, 20, 30]) - 20.0).abs() < f64::EPSILON);
        assert!((mean(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn median_odd() {
        assert!((median(&[10, 20, 30]) - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn median_even() {
        assert!((median(&[10, 20, 30, 40]) - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stddev_zero_for_constant() {
        assert!((stddev(&[5, 5, 5, 5], 5.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stddev_nonzero() {
        let sd = stddev(&[2, 4, 4, 4, 5, 5, 7, 9], 5.0);
        // Population std dev is 2.0, sample std dev is ~2.138
        assert!(sd > 2.0 && sd < 2.2);
    }

    #[test]
    fn outlier_removal() {
        let samples = vec![10, 11, 12, 13, 14, 15, 100];
        let cleaned = remove_outliers_iqr(&samples);
        // 100 should be removed as outlier
        assert!(!cleaned.contains(&100));
        assert!(cleaned.contains(&10));
        assert!(cleaned.contains(&15));
    }

    #[test]
    fn outlier_removal_small_sample() {
        let samples = vec![1, 2, 3];
        let cleaned = remove_outliers_iqr(&samples);
        assert_eq!(cleaned.len(), 3);
    }

    #[test]
    fn benchmark_result_ops_per_sec() {
        let r = result_from_samples("test", &[1000, 1000, 1000]);
        // 1000 ns/iter = 1_000_000 ops/sec
        assert!((r.ops_per_sec() - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn benchmark_result_throughput() {
        let r = result_from_samples("test", &[1_000_000_000]); // 1 second per iter
        let mbps = r.throughput_mbps(1024 * 1024); // 1 MB per iter
        assert!((mbps - 1.0).abs() < 0.01);
    }

    #[test]
    fn comparison_regression() {
        let baseline = result_from_samples("v1", &[100, 100, 100]);
        let candidate = result_from_samples("v2", &[200, 200, 200]);
        let cmp = compare_results(&baseline, &candidate);
        assert!(cmp.is_regression);
        assert!((cmp.change_percentage - 100.0).abs() < 0.01);
    }

    #[test]
    fn comparison_speedup() {
        let baseline = result_from_samples("v1", &[200, 200, 200]);
        let candidate = result_from_samples("v2", &[100, 100, 100]);
        let cmp = compare_results(&baseline, &candidate);
        assert!(!cmp.is_regression);
        assert!((cmp.change_percentage - (-50.0)).abs() < 0.01);
    }

    #[test]
    fn table_formatting() {
        let results = vec![
            result_from_samples("sort_100", &[500, 600, 550]),
            result_from_samples("sort_1000", &[5000, 6000, 5500]),
        ];
        let table = format_table(&results);
        assert!(table.contains("sort_100"));
        assert!(table.contains("sort_1000"));
        assert!(table.contains("Mean (ns)"));
    }

    #[test]
    fn bench_actually_runs() {
        let mut counter = 0u64;
        let result = bench("counter", 50, || {
            counter += 1;
        });
        assert_eq!(result.iterations, 50);
        // warmup(5) + 50 measure = 55 total calls
        assert!(counter >= 50);
        assert!(result.mean_ns < 1_000_000_000.0); // < 1 sec
    }

    #[test]
    fn result_display() {
        let r = result_from_samples("example", &[100, 200, 150]);
        let s = format!("{r}");
        assert!(s.contains("example"));
        assert!(s.contains("ns/iter"));
    }

    #[test]
    fn comparison_display() {
        let baseline = result_from_samples("v1", &[100]);
        let candidate = result_from_samples("v2", &[150]);
        let cmp = compare_results(&baseline, &candidate);
        let s = format!("{cmp}");
        assert!(s.contains("v1"));
        assert!(s.contains("v2"));
        assert!(s.contains("regression") || s.contains("speedup"));
    }
}
