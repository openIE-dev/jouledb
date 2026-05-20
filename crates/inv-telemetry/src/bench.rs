use std::time::{Duration, Instant};

use serde::Serialize;

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Name of the benchmark.
    pub name: String,
    /// Number of iterations to execute.
    pub iterations: u64,
    /// Optional warmup iterations (not included in measurements).
    pub warmup: u64,
}

impl BenchmarkConfig {
    /// Create a benchmark config with default warmup (10% of iterations).
    pub fn new(name: &str, iterations: u64) -> Self {
        Self {
            name: name.to_string(),
            iterations,
            warmup: iterations / 10,
        }
    }

    /// Set a custom warmup count.
    pub fn with_warmup(mut self, warmup: u64) -> Self {
        self.warmup = warmup;
        self
    }
}

/// Result of a benchmark run.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: u64,
    pub total_duration_ns: u128,
    pub min_ns: u128,
    pub max_ns: u128,
    pub mean_ns: f64,
    pub std_dev_ns: f64,
    pub p50_ns: u128,
    pub p95_ns: u128,
    pub p99_ns: u128,
    /// Operations per second.
    pub throughput: f64,
}

impl BenchmarkResult {
    /// Human-readable report of the benchmark.
    pub fn report(&self) -> String {
        format!(
            "{}: {} iterations in {:.2}ms\n  min={:.2}us  max={:.2}us  mean={:.2}us  stddev={:.2}us\n  p50={:.2}us  p95={:.2}us  p99={:.2}us\n  throughput={:.0} ops/sec",
            self.name,
            self.iterations,
            self.total_duration_ns as f64 / 1_000_000.0,
            self.min_ns as f64 / 1_000.0,
            self.max_ns as f64 / 1_000.0,
            self.mean_ns / 1_000.0,
            self.std_dev_ns / 1_000.0,
            self.p50_ns as f64 / 1_000.0,
            self.p95_ns as f64 / 1_000.0,
            self.p99_ns as f64 / 1_000.0,
            self.throughput,
        )
    }
}

/// A lightweight benchmark runner.
///
/// Runs a closure repeatedly, measures timing, and computes statistics.
/// Not a replacement for criterion — this is for quick in-process
/// benchmarks and CI regression checks.
pub struct BenchmarkRunner {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Run a benchmark and record the result.
    ///
    /// The provided closure is called `config.iterations` times (plus
    /// warmup). Each invocation is timed individually.
    pub fn run<F>(&mut self, config: &BenchmarkConfig, mut func: F)
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..config.warmup {
            func();
        }

        let mut durations = Vec::with_capacity(config.iterations as usize);
        let total_start = Instant::now();

        for _ in 0..config.iterations {
            let start = Instant::now();
            func();
            durations.push(start.elapsed());
        }

        let total_duration = total_start.elapsed();
        let result =
            Self::compute_stats(&config.name, config.iterations, &durations, total_duration);
        self.results.push(result);
    }

    /// All benchmark results collected so far.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Generate a combined report of all benchmarks.
    pub fn report(&self) -> String {
        self.results
            .iter()
            .map(|r| r.report())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Export all results as a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.results)
            .unwrap_or(serde_json::json!({"error": "serialization failed"}))
    }

    /// Clear all results.
    pub fn clear(&mut self) {
        self.results.clear();
    }

    /// Compute statistics from a vector of durations.
    fn compute_stats(
        name: &str,
        iterations: u64,
        durations: &[Duration],
        total: Duration,
    ) -> BenchmarkResult {
        let mut nanos: Vec<u128> = durations.iter().map(|d| d.as_nanos()).collect();
        nanos.sort();

        let count = nanos.len();
        let min_ns = nanos.first().copied().unwrap_or(0);
        let max_ns = nanos.last().copied().unwrap_or(0);
        let sum: u128 = nanos.iter().sum();
        let mean_ns = if count > 0 {
            sum as f64 / count as f64
        } else {
            0.0
        };

        let variance = if count > 1 {
            nanos
                .iter()
                .map(|&n| (n as f64 - mean_ns).powi(2))
                .sum::<f64>()
                / (count - 1) as f64
        } else {
            0.0
        };
        let std_dev_ns = variance.sqrt();

        let p50_ns = Self::percentile(&nanos, 0.50);
        let p95_ns = Self::percentile(&nanos, 0.95);
        let p99_ns = Self::percentile(&nanos, 0.99);

        let throughput = if total.as_secs_f64() > 0.0 {
            iterations as f64 / total.as_secs_f64()
        } else {
            0.0
        };

        BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_duration_ns: total.as_nanos(),
            min_ns,
            max_ns,
            mean_ns,
            std_dev_ns,
            p50_ns,
            p95_ns,
            p99_ns,
            throughput,
        }
    }

    /// Compute a percentile from a sorted array.
    fn percentile(sorted: &[u128], p: f64) -> u128 {
        if sorted.is_empty() {
            return 0;
        }
        let idx = ((p * sorted.len() as f64) as usize).min(sorted.len() - 1);
        sorted[idx]
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_config_defaults() {
        let cfg = BenchmarkConfig::new("test", 1000);
        assert_eq!(cfg.name, "test");
        assert_eq!(cfg.iterations, 1000);
        assert_eq!(cfg.warmup, 100);
    }

    #[test]
    fn benchmark_config_custom_warmup() {
        let cfg = BenchmarkConfig::new("test", 1000).with_warmup(50);
        assert_eq!(cfg.warmup, 50);
    }

    #[test]
    fn run_simple_benchmark() {
        let mut runner = BenchmarkRunner::new();
        let cfg = BenchmarkConfig::new("noop", 100).with_warmup(0);
        runner.run(&cfg, || {});

        assert_eq!(runner.results().len(), 1);
        let result = &runner.results()[0];
        assert_eq!(result.name, "noop");
        assert_eq!(result.iterations, 100);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn statistics_computed() {
        let mut runner = BenchmarkRunner::new();
        let cfg = BenchmarkConfig::new("sleep", 10).with_warmup(0);
        runner.run(&cfg, || {
            std::thread::sleep(Duration::from_micros(100));
        });

        let result = &runner.results()[0];
        assert!(result.min_ns > 0);
        assert!(result.max_ns >= result.min_ns);
        assert!(result.mean_ns >= result.min_ns as f64);
        assert!(result.p50_ns >= result.min_ns);
    }

    #[test]
    fn report_format() {
        let mut runner = BenchmarkRunner::new();
        let cfg = BenchmarkConfig::new("test_op", 50).with_warmup(0);
        runner.run(&cfg, || {});

        let report = runner.report();
        assert!(report.contains("test_op"));
        assert!(report.contains("50 iterations"));
        assert!(report.contains("ops/sec"));
    }

    #[test]
    fn multiple_benchmarks() {
        let mut runner = BenchmarkRunner::new();
        runner.run(&BenchmarkConfig::new("a", 10).with_warmup(0), || {});
        runner.run(&BenchmarkConfig::new("b", 20).with_warmup(0), || {});

        assert_eq!(runner.results().len(), 2);
        assert_eq!(runner.results()[0].name, "a");
        assert_eq!(runner.results()[1].name, "b");
    }

    #[test]
    fn to_json_output() {
        let mut runner = BenchmarkRunner::new();
        runner.run(&BenchmarkConfig::new("json_test", 10).with_warmup(0), || {});

        let json = runner.to_json();
        assert!(json.is_array());
        assert_eq!(json[0]["name"], "json_test");
    }

    #[test]
    fn clear_removes_results() {
        let mut runner = BenchmarkRunner::new();
        runner.run(&BenchmarkConfig::new("x", 10).with_warmup(0), || {});
        assert_eq!(runner.results().len(), 1);
        runner.clear();
        assert_eq!(runner.results().len(), 0);
    }

    #[test]
    fn default_creates_empty() {
        let runner = BenchmarkRunner::default();
        assert!(runner.results().is_empty());
    }

    #[test]
    fn percentile_computation() {
        let sorted: Vec<u128> = (1..=100).collect();
        // Index = (p * len) clamped to len-1: 0.50*100=50 → sorted[50]=51
        assert_eq!(BenchmarkRunner::percentile(&sorted, 0.50), 51);
        assert_eq!(BenchmarkRunner::percentile(&sorted, 0.95), 96);
        assert_eq!(BenchmarkRunner::percentile(&sorted, 0.99), 100);
    }

    #[test]
    fn result_report_string() {
        let result = BenchmarkResult {
            name: "test".to_string(),
            iterations: 1000,
            total_duration_ns: 1_000_000,
            min_ns: 500,
            max_ns: 5000,
            mean_ns: 1000.0,
            std_dev_ns: 200.0,
            p50_ns: 900,
            p95_ns: 2000,
            p99_ns: 4000,
            throughput: 1_000_000.0,
        };

        let report = result.report();
        assert!(report.contains("test: 1000 iterations"));
        assert!(report.contains("throughput=1000000 ops/sec"));
    }
}
