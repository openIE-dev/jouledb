//! Monte Carlo simulation — random sampling, estimation, confidence intervals.
//!
//! Replaces jStat / Monte Carlo JS / simjs with pure Rust.
//! Supports configurable PRNG (xorshift64), expected value estimation,
//! confidence intervals, importance sampling, rejection sampling,
//! convergence tracking, and a pi estimation demo.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for Monte Carlo simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McError {
    /// Zero sample count.
    ZeroSamples,
    /// Empty dataset.
    EmptyData,
    /// Invalid probability (not in [0, 1]).
    InvalidProbability(String),
    /// Convergence not reached.
    NotConverged { iterations: u64, threshold: String },
}

impl fmt::Display for McError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSamples => write!(f, "sample count must be non-zero"),
            Self::EmptyData => write!(f, "dataset is empty"),
            Self::InvalidProbability(p) => write!(f, "invalid probability: {p}"),
            Self::NotConverged { iterations, threshold } => {
                write!(f, "not converged after {iterations} iterations (threshold: {threshold})")
            }
        }
    }
}

impl std::error::Error for McError {}

// ── PRNG (xorshift64) ──────────────────────────────────────────

/// A simple xorshift64 pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create with a seed. Seed must not be zero.
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    /// Next raw u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform f64 in [lo, hi).
    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Standard normal via Box-Muller transform.
    pub fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    }
}

// ── Statistics helpers ──────────────────────────────────────────

/// Summary statistics of a sample.
#[derive(Debug, Clone)]
pub struct SampleStats {
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

/// Compute sample statistics from a slice of values.
pub fn compute_stats(data: &[f64]) -> Result<SampleStats, McError> {
    if data.is_empty() {
        return Err(McError::EmptyData);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = if data.len() == 1 {
        0.0
    } else {
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    };
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in data {
        if v < min { min = v; }
        if v > max { max = v; }
    }
    Ok(SampleStats {
        mean,
        variance,
        std_dev: variance.sqrt(),
        min,
        max,
        count: data.len(),
    })
}

/// Compute a confidence interval for the mean.
/// Uses z-value approximation (1.96 for 95%, 2.576 for 99%).
pub fn confidence_interval(data: &[f64], z: f64) -> Result<(f64, f64), McError> {
    let stats = compute_stats(data)?;
    let margin = z * stats.std_dev / (stats.count as f64).sqrt();
    Ok((stats.mean - margin, stats.mean + margin))
}

// ── Monte Carlo Estimator ───────────────────────────────────────

/// A Monte Carlo estimator that tracks convergence.
#[derive(Debug, Clone)]
pub struct MonteCarloEstimator {
    rng: Xorshift64,
    samples: Vec<f64>,
    running_sum: f64,
    running_sum_sq: f64,
    convergence_history: Vec<f64>,
}

impl MonteCarloEstimator {
    /// Create a new estimator with a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            samples: Vec::new(),
            running_sum: 0.0,
            running_sum_sq: 0.0,
            convergence_history: Vec::new(),
        }
    }

    /// Add a sample value.
    pub fn add_sample(&mut self, value: f64) {
        self.samples.push(value);
        self.running_sum += value;
        self.running_sum_sq += value * value;
        let mean = self.running_sum / self.samples.len() as f64;
        self.convergence_history.push(mean);
    }

    /// Number of samples collected.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Current running mean.
    pub fn current_mean(&self) -> Option<f64> {
        if self.samples.is_empty() {
            None
        } else {
            Some(self.running_sum / self.samples.len() as f64)
        }
    }

    /// Current running variance.
    pub fn current_variance(&self) -> Option<f64> {
        let n = self.samples.len();
        if n < 2 {
            return None;
        }
        let nf = n as f64;
        let mean = self.running_sum / nf;
        Some((self.running_sum_sq / nf - mean * mean) * nf / (nf - 1.0))
    }

    /// Current standard error of the mean.
    pub fn standard_error(&self) -> Option<f64> {
        self.current_variance().map(|v| (v / self.samples.len() as f64).sqrt())
    }

    /// Convergence history (running mean at each step).
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Check if the estimator has converged within a tolerance.
    pub fn has_converged(&self, tolerance: f64) -> bool {
        let n = self.convergence_history.len();
        if n < 10 {
            return false;
        }
        let recent = &self.convergence_history[n - 5..];
        let old = &self.convergence_history[n - 10..n - 5];
        let recent_mean: f64 = recent.iter().sum::<f64>() / 5.0;
        let old_mean: f64 = old.iter().sum::<f64>() / 5.0;
        (recent_mean - old_mean).abs() < tolerance
    }

    /// 95% confidence interval for the mean.
    pub fn confidence_interval_95(&self) -> Option<(f64, f64)> {
        if self.samples.len() < 2 {
            return None;
        }
        confidence_interval(&self.samples, 1.96).ok()
    }

    /// 99% confidence interval for the mean.
    pub fn confidence_interval_99(&self) -> Option<(f64, f64)> {
        if self.samples.len() < 2 {
            return None;
        }
        confidence_interval(&self.samples, 2.576).ok()
    }

    /// Get access to the PRNG.
    pub fn rng_mut(&mut self) -> &mut Xorshift64 {
        &mut self.rng
    }

    /// All collected samples.
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }

    /// Reset the estimator (keeps seed state).
    pub fn reset(&mut self) {
        self.samples.clear();
        self.running_sum = 0.0;
        self.running_sum_sq = 0.0;
        self.convergence_history.clear();
    }
}

// ── Importance Sampling ─────────────────────────────────────────

/// Result of importance sampling.
#[derive(Debug, Clone)]
pub struct ImportanceSampleResult {
    pub estimate: f64,
    pub effective_sample_size: f64,
    pub weights: Vec<f64>,
}

/// Perform importance sampling.
///
/// `target_samples` are samples from the proposal distribution.
/// `target_weights` are the importance weights (target_pdf / proposal_pdf).
pub fn importance_sampling(
    target_samples: &[f64],
    target_weights: &[f64],
) -> Result<ImportanceSampleResult, McError> {
    if target_samples.is_empty() || target_weights.is_empty() {
        return Err(McError::EmptyData);
    }
    if target_samples.len() != target_weights.len() {
        return Err(McError::EmptyData);
    }

    let weight_sum: f64 = target_weights.iter().sum();
    if weight_sum == 0.0 {
        return Err(McError::EmptyData);
    }

    let normalized: Vec<f64> = target_weights.iter().map(|w| w / weight_sum).collect();
    let estimate: f64 = target_samples.iter()
        .zip(normalized.iter())
        .map(|(s, w)| s * w)
        .sum();

    let sum_sq: f64 = normalized.iter().map(|w| w * w).sum();
    let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

    Ok(ImportanceSampleResult {
        estimate,
        effective_sample_size: ess,
        weights: normalized,
    })
}

// ── Rejection Sampling ──────────────────────────────────────────

/// Perform rejection sampling.
///
/// Samples uniformly in [lo, hi] and accepts if `accept_fn(x)` returns true.
/// Returns accepted samples.
pub fn rejection_sampling(
    rng: &mut Xorshift64,
    lo: f64,
    hi: f64,
    n_attempts: usize,
    accept_fn: impl Fn(f64) -> bool,
) -> Vec<f64> {
    let mut accepted = Vec::new();
    for _ in 0..n_attempts {
        let x = rng.next_range(lo, hi);
        if accept_fn(x) {
            accepted.push(x);
        }
    }
    accepted
}

// ── Pi Estimation ───────────────────────────────────────────────

/// Estimate pi using Monte Carlo (dart-throwing method).
///
/// Throw `n` darts at a unit square. Count those inside the inscribed circle.
/// pi ~= 4 * (inside / total).
pub fn estimate_pi(rng: &mut Xorshift64, n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut inside = 0u64;
    for _ in 0..n {
        let x = rng.next_f64() * 2.0 - 1.0;
        let y = rng.next_f64() * 2.0 - 1.0;
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
    }
    4.0 * inside as f64 / n as f64
}

/// Estimate pi with convergence tracking.
pub fn estimate_pi_with_convergence(rng: &mut Xorshift64, n: u64, interval: u64) -> Vec<(u64, f64)> {
    let mut inside = 0u64;
    let mut history = Vec::new();
    for i in 1..=n {
        let x = rng.next_f64() * 2.0 - 1.0;
        let y = rng.next_f64() * 2.0 - 1.0;
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
        if i % interval == 0 || i == n {
            history.push((i, 4.0 * inside as f64 / i as f64));
        }
    }
    history
}

// ── Integration ─────────────────────────────────────────────────

/// Estimate the integral of f(x) over [a, b] using Monte Carlo integration.
pub fn integrate(
    rng: &mut Xorshift64,
    a: f64,
    b: f64,
    n: u64,
    f: impl Fn(f64) -> f64,
) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for _ in 0..n {
        let x = rng.next_range(a, b);
        sum += f(x);
    }
    (b - a) * sum / n as f64
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_deterministic() {
        let mut r1 = Xorshift64::new(42);
        let mut r2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_xorshift_different_seeds() {
        let mut r1 = Xorshift64::new(42);
        let mut r2 = Xorshift64::new(99);
        let mut same = true;
        for _ in 0..10 {
            if r1.next_u64() != r2.next_u64() {
                same = false;
                break;
            }
        }
        assert!(!same);
    }

    #[test]
    fn test_xorshift_zero_seed_handled() {
        let mut r = Xorshift64::new(0);
        // Should not get stuck at zero.
        assert_ne!(r.next_u64(), 0);
    }

    #[test]
    fn test_next_f64_range() {
        let mut r = Xorshift64::new(42);
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_next_range() {
        let mut r = Xorshift64::new(42);
        for _ in 0..1000 {
            let v = r.next_range(5.0, 10.0);
            assert!(v >= 5.0 && v < 10.0);
        }
    }

    #[test]
    fn test_next_normal_reasonable() {
        let mut r = Xorshift64::new(42);
        let mut sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            sum += r.next_normal(0.0, 1.0);
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "mean was {mean}");
    }

    #[test]
    fn test_compute_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_stats(&data).unwrap();
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert_eq!(stats.count, 5);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_empty() {
        assert!(compute_stats(&[]).is_err());
    }

    #[test]
    fn test_compute_stats_single() {
        let stats = compute_stats(&[7.0]).unwrap();
        assert!((stats.mean - 7.0).abs() < 1e-10);
        assert!((stats.variance - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_interval() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (lo, hi) = confidence_interval(&data, 1.96).unwrap();
        let stats = compute_stats(&data).unwrap();
        assert!(lo < stats.mean);
        assert!(hi > stats.mean);
        assert!(hi - lo > 0.0);
    }

    #[test]
    fn test_estimator_running_mean() {
        let mut est = MonteCarloEstimator::new(42);
        est.add_sample(10.0);
        est.add_sample(20.0);
        assert!((est.current_mean().unwrap() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimator_sample_count() {
        let mut est = MonteCarloEstimator::new(42);
        assert_eq!(est.sample_count(), 0);
        est.add_sample(1.0);
        est.add_sample(2.0);
        assert_eq!(est.sample_count(), 2);
    }

    #[test]
    fn test_estimator_convergence() {
        let mut est = MonteCarloEstimator::new(42);
        // Feed the same value repeatedly — should converge.
        for _ in 0..20 {
            est.add_sample(5.0);
        }
        assert!(est.has_converged(0.01));
    }

    #[test]
    fn test_estimator_not_converged_with_few_samples() {
        let mut est = MonteCarloEstimator::new(42);
        est.add_sample(1.0);
        assert!(!est.has_converged(0.01));
    }

    #[test]
    fn test_estimator_ci95() {
        let mut est = MonteCarloEstimator::new(42);
        for i in 0..100 {
            est.add_sample(i as f64);
        }
        let ci = est.confidence_interval_95().unwrap();
        assert!(ci.0 < ci.1);
    }

    #[test]
    fn test_estimator_ci99() {
        let mut est = MonteCarloEstimator::new(42);
        for i in 0..100 {
            est.add_sample(i as f64);
        }
        let ci99 = est.confidence_interval_99().unwrap();
        let ci95 = est.confidence_interval_95().unwrap();
        // 99% CI should be wider than 95%.
        assert!(ci99.1 - ci99.0 > ci95.1 - ci95.0);
    }

    #[test]
    fn test_estimator_reset() {
        let mut est = MonteCarloEstimator::new(42);
        est.add_sample(1.0);
        est.reset();
        assert_eq!(est.sample_count(), 0);
        assert!(est.current_mean().is_none());
    }

    #[test]
    fn test_importance_sampling() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = importance_sampling(&samples, &weights).unwrap();
        assert!((result.estimate - 3.0).abs() < 1e-10);
        assert!(result.effective_sample_size > 0.0);
    }

    #[test]
    fn test_importance_sampling_unequal_weights() {
        let samples = vec![0.0, 10.0];
        let weights = vec![1.0, 3.0];
        let result = importance_sampling(&samples, &weights).unwrap();
        // Weighted mean: (0*0.25 + 10*0.75) = 7.5.
        assert!((result.estimate - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_importance_sampling_empty() {
        assert!(importance_sampling(&[], &[]).is_err());
    }

    #[test]
    fn test_rejection_sampling() {
        let mut rng = Xorshift64::new(42);
        let accepted = rejection_sampling(&mut rng, 0.0, 1.0, 10_000, |x| x > 0.5);
        assert!(!accepted.is_empty());
        for &v in &accepted {
            assert!(v > 0.5);
        }
    }

    #[test]
    fn test_estimate_pi() {
        let mut rng = Xorshift64::new(42);
        let pi = estimate_pi(&mut rng, 100_000);
        assert!((pi - std::f64::consts::PI).abs() < 0.1, "pi estimate was {pi}");
    }

    #[test]
    fn test_estimate_pi_zero() {
        let mut rng = Xorshift64::new(42);
        assert_eq!(estimate_pi(&mut rng, 0), 0.0);
    }

    #[test]
    fn test_estimate_pi_with_convergence() {
        let mut rng = Xorshift64::new(42);
        let hist = estimate_pi_with_convergence(&mut rng, 10_000, 1000);
        assert!(!hist.is_empty());
        // Last entry should be close to pi.
        let (_, last_pi) = hist.last().unwrap();
        assert!((last_pi - std::f64::consts::PI).abs() < 0.2);
        // Should show monotonic sample counts.
        for (i, &(n, _)) in hist.iter().enumerate() {
            if i > 0 {
                assert!(n > hist[i - 1].0);
            }
        }
    }

    #[test]
    fn test_integrate_constant() {
        let mut rng = Xorshift64::new(42);
        let result = integrate(&mut rng, 0.0, 1.0, 10_000, |_| 5.0);
        assert!((result - 5.0).abs() < 0.1, "integral was {result}");
    }

    #[test]
    fn test_integrate_linear() {
        let mut rng = Xorshift64::new(42);
        // Integral of x from 0 to 1 = 0.5.
        let result = integrate(&mut rng, 0.0, 1.0, 100_000, |x| x);
        assert!((result - 0.5).abs() < 0.05, "integral was {result}");
    }

    #[test]
    fn test_standard_error_decreases() {
        let mut est = MonteCarloEstimator::new(42);
        for _ in 0..50 {
            let v = est.rng_mut().next_normal(0.0, 1.0);
            est.add_sample(v);
        }
        let se50 = est.standard_error().unwrap();
        for _ in 0..450 {
            let v = est.rng_mut().next_normal(0.0, 1.0);
            est.add_sample(v);
        }
        let se500 = est.standard_error().unwrap();
        assert!(se500 < se50, "SE should decrease: {se500} vs {se50}");
    }

    #[test]
    fn test_convergence_history_length() {
        let mut est = MonteCarloEstimator::new(42);
        for i in 0..25 {
            est.add_sample(i as f64);
        }
        assert_eq!(est.convergence_history().len(), 25);
    }
}
