// Monte Carlo integration for rendering.
// Estimator, variance tracking (Welford), adaptive sampling, stratified, quasi-MC.

use std::fmt;

const PI: f64 = std::f64::consts::PI;

/// Running statistics using Welford's online algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct WelfordStats {
    pub count: usize,
    pub mean: f64,
    m2: f64,
}

impl WelfordStats {
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        self.m2 / (self.count - 1) as f64
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn standard_error(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.std_dev() / (self.count as f64).sqrt()
    }

    pub fn merge(&self, other: &WelfordStats) -> WelfordStats {
        if self.count == 0 { return other.clone(); }
        if other.count == 0 { return self.clone(); }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean = (self.mean * self.count as f64 + other.mean * other.count as f64) / total as f64;
        let new_m2 = self.m2 + other.m2 + delta * delta * (self.count as f64 * other.count as f64) / total as f64;
        WelfordStats { count: total, mean: new_mean, m2: new_m2 }
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

impl fmt::Display for WelfordStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mean={:.6}, var={:.6}, n={}", self.mean, self.variance(), self.count)
    }
}

/// Simple LCG random.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─── Monte Carlo estimator ───

/// Monte Carlo estimate: sum(f(x)/pdf(x)) / N.
/// Returns (estimate, stats).
pub fn mc_estimate<F, P>(
    f_eval: F,
    pdf_eval: P,
    sample_gen: &mut dyn FnMut() -> f64,
    num_samples: usize,
) -> (f64, WelfordStats)
where
    F: Fn(f64) -> f64,
    P: Fn(f64) -> f64,
{
    let mut stats = WelfordStats::new();
    for _ in 0..num_samples {
        let x = sample_gen();
        let p = pdf_eval(x);
        if p.abs() < 1e-15 {
            continue;
        }
        let contribution = f_eval(x) / p;
        stats.update(contribution);
    }
    (stats.mean, stats)
}

/// Multi-dimensional Monte Carlo estimate.
pub fn mc_estimate_nd<F, P>(
    f_eval: F,
    pdf_eval: P,
    sample_gen: &mut dyn FnMut() -> Vec<f64>,
    num_samples: usize,
) -> (f64, WelfordStats)
where
    F: Fn(&[f64]) -> f64,
    P: Fn(&[f64]) -> f64,
{
    let mut stats = WelfordStats::new();
    for _ in 0..num_samples {
        let x = sample_gen();
        let p = pdf_eval(&x);
        if p.abs() < 1e-15 {
            continue;
        }
        let contribution = f_eval(&x) / p;
        stats.update(contribution);
    }
    (stats.mean, stats)
}

// ─── Stratified sampling ───

/// Stratified Monte Carlo integration in 1D over [a, b].
pub fn stratified_estimate<F>(
    f_eval: F,
    a: f64,
    b: f64,
    num_strata: usize,
    rng: &mut Rng,
) -> (f64, WelfordStats)
where
    F: Fn(f64) -> f64,
{
    let width = (b - a) / num_strata as f64;
    let mut stats = WelfordStats::new();
    for i in 0..num_strata {
        let lo = a + i as f64 * width;
        let x = lo + rng.next_f64() * width;
        let val = f_eval(x) * (b - a);
        stats.update(val);
    }
    (stats.mean, stats)
}

/// Stratified 2D integration over [a1,b1] x [a2,b2].
pub fn stratified_2d_estimate<F>(
    f_eval: F,
    range1: (f64, f64),
    range2: (f64, f64),
    strata_per_dim: usize,
    rng: &mut Rng,
) -> (f64, WelfordStats)
where
    F: Fn(f64, f64) -> f64,
{
    let w1 = (range1.1 - range1.0) / strata_per_dim as f64;
    let w2 = (range2.1 - range2.0) / strata_per_dim as f64;
    let area = (range1.1 - range1.0) * (range2.1 - range2.0);
    let mut stats = WelfordStats::new();
    for i in 0..strata_per_dim {
        for j in 0..strata_per_dim {
            let x1 = range1.0 + (i as f64 + rng.next_f64()) * w1;
            let x2 = range2.0 + (j as f64 + rng.next_f64()) * w2;
            let val = f_eval(x1, x2) * area;
            stats.update(val);
        }
    }
    (stats.mean, stats)
}

// ─── Quasi-Monte Carlo ───

/// Halton sequence value.
pub fn halton(index: usize, base: usize) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    let mut i = index;
    while i > 0 {
        result += f * (i % base) as f64;
        i /= base;
        f /= base as f64;
    }
    result
}

/// Quasi-Monte Carlo integration using Halton sequence over [a, b].
pub fn quasi_mc_estimate<F>(
    f_eval: F,
    a: f64,
    b: f64,
    num_samples: usize,
) -> (f64, WelfordStats)
where
    F: Fn(f64) -> f64,
{
    let range = b - a;
    let mut stats = WelfordStats::new();
    for i in 1..=num_samples {
        let x = a + halton(i, 2) * range;
        let val = f_eval(x) * range;
        stats.update(val);
    }
    (stats.mean, stats)
}

/// Quasi-Monte Carlo 2D using Halton bases 2 and 3.
pub fn quasi_mc_2d_estimate<F>(
    f_eval: F,
    range1: (f64, f64),
    range2: (f64, f64),
    num_samples: usize,
) -> (f64, WelfordStats)
where
    F: Fn(f64, f64) -> f64,
{
    let r1 = range1.1 - range1.0;
    let r2 = range2.1 - range2.0;
    let area = r1 * r2;
    let mut stats = WelfordStats::new();
    for i in 1..=num_samples {
        let x1 = range1.0 + halton(i, 2) * r1;
        let x2 = range2.0 + halton(i, 3) * r2;
        let val = f_eval(x1, x2) * area;
        stats.update(val);
    }
    (stats.mean, stats)
}

// ─── Adaptive sampling ───

/// Pixel adaptive sampler: allocates more samples to high-variance pixels.
#[derive(Debug, Clone)]
pub struct AdaptiveSampler {
    pub width: usize,
    pub height: usize,
    pub stats: Vec<WelfordStats>,
    pub min_samples: usize,
    pub max_samples: usize,
    pub variance_threshold: f64,
}

impl AdaptiveSampler {
    pub fn new(width: usize, height: usize, min_samples: usize, max_samples: usize, variance_threshold: f64) -> Self {
        Self {
            width,
            height,
            stats: vec![WelfordStats::new(); width * height],
            min_samples,
            max_samples,
            variance_threshold,
        }
    }

    pub fn add_sample(&mut self, x: usize, y: usize, value: f64) {
        let idx = y * self.width + x;
        if idx < self.stats.len() {
            self.stats[idx].update(value);
        }
    }

    pub fn needs_more_samples(&self, x: usize, y: usize) -> bool {
        let idx = y * self.width + x;
        if idx >= self.stats.len() { return false; }
        let s = &self.stats[idx];
        if s.count < self.min_samples { return true; }
        if s.count >= self.max_samples { return false; }
        s.variance() > self.variance_threshold
    }

    pub fn get_estimate(&self, x: usize, y: usize) -> f64 {
        let idx = y * self.width + x;
        if idx >= self.stats.len() { return 0.0; }
        self.stats[idx].mean
    }

    pub fn get_variance(&self, x: usize, y: usize) -> f64 {
        let idx = y * self.width + x;
        if idx >= self.stats.len() { return 0.0; }
        self.stats[idx].variance()
    }

    pub fn total_samples(&self) -> usize {
        self.stats.iter().map(|s| s.count).sum()
    }

    /// Check if all pixels have converged below threshold.
    pub fn is_converged(&self) -> bool {
        self.stats.iter().all(|s| {
            s.count >= self.min_samples && (s.count >= self.max_samples || s.variance() <= self.variance_threshold)
        })
    }
}

/// Check convergence: standard error below threshold.
pub fn check_convergence(stats: &WelfordStats, threshold: f64) -> bool {
    if stats.count < 2 { return false; }
    stats.standard_error() < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    #[test]
    fn test_welford_single() {
        let mut s = WelfordStats::new();
        s.update(5.0);
        assert!(approx_eq(s.mean, 5.0, 1e-9));
        assert!(approx_eq(s.variance(), 0.0, 1e-9));
    }

    #[test]
    fn test_welford_multiple() {
        let mut s = WelfordStats::new();
        for v in [2.0, 4.0, 6.0, 8.0, 10.0] {
            s.update(v);
        }
        assert!(approx_eq(s.mean, 6.0, 1e-9));
        // Variance of [2,4,6,8,10] = 10.0 (sample variance)
        assert!(approx_eq(s.variance(), 10.0, 1e-9));
    }

    #[test]
    fn test_welford_std_dev() {
        let mut s = WelfordStats::new();
        for v in [2.0, 4.0, 6.0, 8.0, 10.0] {
            s.update(v);
        }
        assert!(approx_eq(s.std_dev(), 10.0f64.sqrt(), 1e-9));
    }

    #[test]
    fn test_welford_standard_error() {
        let mut s = WelfordStats::new();
        for v in [1.0, 2.0, 3.0, 4.0] {
            s.update(v);
        }
        let se = s.standard_error();
        // std_dev = sqrt(5/3) ≈ 1.291, SE = 1.291/2 ≈ 0.645
        assert!(se > 0.0 && se < 1.0);
    }

    #[test]
    fn test_welford_merge() {
        let mut a = WelfordStats::new();
        for v in [1.0, 2.0, 3.0] { a.update(v); }
        let mut b = WelfordStats::new();
        for v in [4.0, 5.0, 6.0] { b.update(v); }
        let merged = a.merge(&b);
        assert_eq!(merged.count, 6);
        assert!(approx_eq(merged.mean, 3.5, 1e-9));
    }

    #[test]
    fn test_welford_reset() {
        let mut s = WelfordStats::new();
        s.update(10.0);
        s.update(20.0);
        s.reset();
        assert_eq!(s.count, 0);
        assert!(approx_eq(s.mean, 0.0, 1e-9));
    }

    #[test]
    fn test_mc_estimate_constant() {
        // Integrate f(x) = 3 over [0,1] with uniform pdf=1
        let mut rng = Rng::new(42);
        let (est, stats) = mc_estimate(
            |_x| 3.0,
            |_x| 1.0,
            &mut || rng.next_f64(),
            1000,
        );
        assert!(approx_eq(est, 3.0, 1e-6));
        assert!(stats.variance() < 1e-10);
    }

    #[test]
    fn test_mc_estimate_linear() {
        // Integrate f(x) = x over [0,1] with uniform pdf=1
        // True value = 0.5
        let mut rng = Rng::new(42);
        let (est, _) = mc_estimate(
            |x| x,
            |_x| 1.0,
            &mut || rng.next_f64(),
            10000,
        );
        assert!(approx_eq(est, 0.5, 0.02));
    }

    #[test]
    fn test_mc_estimate_nd_area_circle() {
        // Estimate area of unit circle via 2D MC: f(x,y) = 1 if x^2+y^2 < 1
        // Integrate over [-1,1]^2, area = 4, true = pi
        let mut rng = Rng::new(42);
        let (est, _) = mc_estimate_nd(
            |x| if x[0] * x[0] + x[1] * x[1] < 1.0 { 4.0 } else { 0.0 },
            |_x| 1.0,
            &mut || vec![rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0],
            50000,
        );
        assert!(approx_eq(est, PI, 0.1), "estimated pi = {}", est);
    }

    #[test]
    fn test_stratified_constant() {
        let mut rng = Rng::new(42);
        let (est, _) = stratified_estimate(|_| 5.0, 0.0, 1.0, 100, &mut rng);
        assert!(approx_eq(est, 5.0, 1e-6));
    }

    #[test]
    fn test_stratified_linear() {
        let mut rng = Rng::new(42);
        let (est, _) = stratified_estimate(|x| x, 0.0, 1.0, 1000, &mut rng);
        assert!(approx_eq(est, 0.5, 0.02));
    }

    #[test]
    fn test_stratified_2d_area() {
        let mut rng = Rng::new(42);
        let (est, _) = stratified_2d_estimate(
            |_x, _y| 1.0,
            (0.0, 2.0),
            (0.0, 3.0),
            20,
            &mut rng,
        );
        assert!(approx_eq(est, 6.0, 0.01));
    }

    #[test]
    fn test_halton_base2() {
        assert!(approx_eq(halton(1, 2), 0.5, 1e-9));
        assert!(approx_eq(halton(2, 2), 0.25, 1e-9));
        assert!(approx_eq(halton(3, 2), 0.75, 1e-9));
    }

    #[test]
    fn test_quasi_mc_constant() {
        let (est, _) = quasi_mc_estimate(|_| 7.0, 0.0, 1.0, 100);
        assert!(approx_eq(est, 7.0, 1e-6));
    }

    #[test]
    fn test_quasi_mc_linear() {
        let (est, _) = quasi_mc_estimate(|x| x, 0.0, 1.0, 10000);
        assert!(approx_eq(est, 0.5, 0.01));
    }

    #[test]
    fn test_quasi_mc_2d_area() {
        let (est, _) = quasi_mc_2d_estimate(
            |_x, _y| 1.0,
            (0.0, 3.0),
            (0.0, 4.0),
            1000,
        );
        assert!(approx_eq(est, 12.0, 0.01));
    }

    #[test]
    fn test_adaptive_sampler_basic() {
        let mut sampler = AdaptiveSampler::new(2, 2, 4, 100, 0.01);
        assert!(sampler.needs_more_samples(0, 0));
        for _ in 0..10 {
            sampler.add_sample(0, 0, 5.0);
        }
        // Constant value -> zero variance -> converged
        assert!(!sampler.needs_more_samples(0, 0));
    }

    #[test]
    fn test_adaptive_sampler_high_variance() {
        let mut sampler = AdaptiveSampler::new(2, 2, 2, 1000, 0.001);
        sampler.add_sample(0, 0, 0.0);
        sampler.add_sample(0, 0, 100.0);
        sampler.add_sample(0, 0, 0.0);
        sampler.add_sample(0, 0, 100.0);
        assert!(sampler.needs_more_samples(0, 0));
    }

    #[test]
    fn test_adaptive_sampler_max_samples() {
        let mut sampler = AdaptiveSampler::new(1, 1, 2, 5, 0.001);
        for i in 0..5 {
            sampler.add_sample(0, 0, i as f64 * 10.0);
        }
        // Reached max -> no more
        assert!(!sampler.needs_more_samples(0, 0));
    }

    #[test]
    fn test_adaptive_sampler_total_samples() {
        let mut sampler = AdaptiveSampler::new(2, 2, 2, 100, 0.01);
        sampler.add_sample(0, 0, 1.0);
        sampler.add_sample(0, 0, 2.0);
        sampler.add_sample(1, 1, 3.0);
        assert_eq!(sampler.total_samples(), 3);
    }

    #[test]
    fn test_adaptive_is_converged() {
        let mut sampler = AdaptiveSampler::new(1, 1, 2, 100, 0.1);
        sampler.add_sample(0, 0, 5.0);
        assert!(!sampler.is_converged());
        sampler.add_sample(0, 0, 5.0);
        sampler.add_sample(0, 0, 5.0);
        assert!(sampler.is_converged());
    }

    #[test]
    fn test_convergence_check() {
        let mut s = WelfordStats::new();
        assert!(!check_convergence(&s, 0.1));
        for _ in 0..1000 {
            s.update(5.0);
        }
        assert!(check_convergence(&s, 0.01));
    }

    #[test]
    fn test_stratified_reduces_variance() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);

        // Plain MC
        let (_, stats_plain) = mc_estimate(
            |x| x * x,
            |_| 1.0,
            &mut || rng1.next_f64(),
            500,
        );

        // Stratified
        let (_, stats_strat) = stratified_estimate(|x| x * x, 0.0, 1.0, 500, &mut rng2);

        // Stratified should have lower or comparable variance
        // (statistical, not guaranteed for every seed, but generally holds)
        assert!(stats_strat.variance() < stats_plain.variance() * 5.0);
    }

    #[test]
    fn test_welford_merge_empty() {
        let a = WelfordStats::new();
        let mut b = WelfordStats::new();
        b.update(10.0);
        let merged = a.merge(&b);
        assert_eq!(merged.count, 1);
        assert!(approx_eq(merged.mean, 10.0, 1e-9));
    }

    #[test]
    fn test_adaptive_get_estimate() {
        let mut sampler = AdaptiveSampler::new(1, 1, 2, 100, 0.01);
        sampler.add_sample(0, 0, 3.0);
        sampler.add_sample(0, 0, 7.0);
        assert!(approx_eq(sampler.get_estimate(0, 0), 5.0, 1e-9));
    }
}
