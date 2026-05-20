//! Random distributions — uniform, normal, exponential, Poisson, binomial,
//! geometric, beta, chi-squared, with seeded sampling, PDF/CDF evaluation, inverse CDF.
//!
//! Pure-Rust random number generation and distribution functions.
//! Uses a simple xorshift64 PRNG for reproducibility.

use std::fmt;

// ── PRNG (xorshift64) ───────────────────────────────────────────

/// Xorshift64 pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new PRNG with the given seed. Seed must be non-zero.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 1 } else { seed };
        Self { state }
    }

    /// Generate the next u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate a uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a uniform f64 in [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Generate a standard normal (mean=0, std=1) using Box-Muller transform.
    pub fn normal_standard(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-15 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return r * theta.cos();
            }
        }
    }

    /// Generate normal(mean, std_dev).
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        mean + std_dev * self.normal_standard()
    }

    /// Generate an exponential(lambda) variate.
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        assert!(lambda > 0.0, "lambda must be positive");
        loop {
            let u = self.next_f64();
            if u > 1e-15 {
                return -u.ln() / lambda;
            }
        }
    }

    /// Generate a Poisson(lambda) variate using Knuth's algorithm.
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda < 30.0 {
            // Knuth's algorithm for small lambda
            let l = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.next_f64();
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // Normal approximation for large lambda
            let v = self.normal(lambda, lambda.sqrt());
            if v < 0.0 { 0 } else { v.round() as u64 }
        }
    }

    /// Generate a binomial(n, p) variate.
    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        if n <= 20 {
            let mut count = 0u64;
            for _ in 0..n {
                if self.next_f64() < p {
                    count += 1;
                }
            }
            count
        } else {
            // Normal approximation
            let mean = n as f64 * p;
            let std = (n as f64 * p * (1.0 - p)).sqrt();
            let v = self.normal(mean, std);
            v.round().clamp(0.0, n as f64) as u64
        }
    }

    /// Generate a geometric(p) variate (number of trials until first success).
    pub fn geometric(&mut self, p: f64) -> u64 {
        assert!(p > 0.0 && p <= 1.0, "p must be in (0, 1]");
        loop {
            let u = self.next_f64();
            if u > 1e-15 {
                return (u.ln() / (1.0 - p).ln()).ceil() as u64;
            }
        }
    }

    /// Generate a beta(alpha, beta) variate using rejection sampling.
    pub fn beta(&mut self, alpha: f64, beta_param: f64) -> f64 {
        // Use the gamma distribution relationship: Beta(a,b) = Ga/(Ga+Gb)
        let x = self.gamma(alpha);
        let y = self.gamma(beta_param);
        if x + y < 1e-15 {
            return 0.5;
        }
        x / (x + y)
    }

    /// Generate a gamma(shape) variate (rate=1). Uses Marsaglia and Tsang's method.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        assert!(shape > 0.0, "shape must be positive");

        if shape < 1.0 {
            // Use the transformation: Gamma(a) = Gamma(a+1) * U^(1/a)
            let g = self.gamma(shape + 1.0);
            let u = self.next_f64();
            return g * u.powf(1.0 / shape);
        }

        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = self.normal_standard();
            let v_val = 1.0 + c * x;
            if v_val <= 0.0 {
                continue;
            }
            let v_val = v_val * v_val * v_val;
            let u = self.next_f64();
            let x2 = x * x;

            if u < 1.0 - 0.0331 * x2 * x2 {
                return d * v_val;
            }
            if u > 1e-15 && u.ln() < 0.5 * x2 + d * (1.0 - v_val + v_val.ln()) {
                return d * v_val;
            }
        }
    }

    /// Generate a chi-squared(k) variate.
    pub fn chi_squared(&mut self, k: u32) -> f64 {
        // Chi-squared(k) = Gamma(k/2, 2) = 2 * Gamma(k/2, 1)
        2.0 * self.gamma(k as f64 / 2.0)
    }

    /// Sample n values from uniform(lo, hi).
    pub fn sample_uniform(&mut self, lo: f64, hi: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }

    /// Sample n values from normal(mean, std_dev).
    pub fn sample_normal(&mut self, mean: f64, std_dev: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal(mean, std_dev)).collect()
    }

    /// Sample n values from exponential(lambda).
    pub fn sample_exponential(&mut self, lambda: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.exponential(lambda)).collect()
    }
}

// ── PDF / CDF evaluation ────────────────────────────────────────

/// Uniform distribution on [a, b].
pub struct Uniform {
    pub a: f64,
    pub b: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a < b);
        Self { a, b }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        if x >= self.a && x <= self.b {
            1.0 / (self.b - self.a)
        } else {
            0.0
        }
    }

    pub fn cdf(&self, x: f64) -> f64 {
        if x < self.a {
            0.0
        } else if x > self.b {
            1.0
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }

    /// Inverse CDF (quantile function).
    pub fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..=1.0).contains(&p));
        self.a + p * (self.b - self.a)
    }
}

impl fmt::Display for Uniform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Uniform({}, {})", self.a, self.b)
    }
}

/// Normal (Gaussian) distribution.
pub struct Normal {
    pub mean: f64,
    pub std_dev: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        assert!(std_dev > 0.0);
        Self { mean, std_dev }
    }

    pub fn standard() -> Self {
        Self { mean: 0.0, std_dev: 1.0 }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        (-0.5 * z * z).exp() / (self.std_dev * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// CDF using error function approximation (Abramowitz & Stegun).
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    }

    /// Inverse CDF using rational approximation (Beasley-Springer-Moro).
    pub fn quantile(&self, p: f64) -> f64 {
        assert!(p > 0.0 && p < 1.0, "p must be in (0, 1)");
        self.mean + self.std_dev * standard_normal_quantile(p)
    }
}

impl fmt::Display for Normal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Normal(mu={}, sigma={})", self.mean, self.std_dev)
    }
}

/// Exponential distribution.
pub struct Exponential {
    pub lambda: f64,
}

impl Exponential {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0);
        Self { lambda }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }

    pub fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.lambda * x).exp()
        }
    }

    pub fn quantile(&self, p: f64) -> f64 {
        assert!(p >= 0.0 && p < 1.0);
        -(1.0 - p).ln() / self.lambda
    }
}

impl fmt::Display for Exponential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Exponential(lambda={})", self.lambda)
    }
}

/// Poisson distribution (discrete).
pub struct Poisson {
    pub lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0);
        Self { lambda }
    }

    /// PMF: P(X=k) = e^{-lambda} * lambda^k / k!
    pub fn pmf(&self, k: u64) -> f64 {
        // Use log to avoid overflow: log(pmf) = -lambda + k*ln(lambda) - ln(k!)
        let log_pmf = -(self.lambda) + k as f64 * self.lambda.ln() - ln_factorial(k);
        log_pmf.exp()
    }

    /// CDF: P(X <= k).
    pub fn cdf(&self, k: u64) -> f64 {
        let mut sum = 0.0;
        for i in 0..=k {
            sum += self.pmf(i);
        }
        sum.min(1.0)
    }
}

impl fmt::Display for Poisson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poisson(lambda={})", self.lambda)
    }
}

/// Binomial distribution (discrete).
pub struct Binomial {
    pub n: u64,
    pub p: f64,
}

impl Binomial {
    pub fn new(n: u64, p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p));
        Self { n, p }
    }

    /// PMF: P(X=k) = C(n,k) * p^k * (1-p)^(n-k).
    pub fn pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return 0.0;
        }
        let log_pmf = ln_binomial_coeff(self.n, k)
            + k as f64 * self.p.ln()
            + (self.n - k) as f64 * (1.0 - self.p).ln();
        log_pmf.exp()
    }

    /// CDF: P(X <= k).
    pub fn cdf(&self, k: u64) -> f64 {
        let mut sum = 0.0;
        let upper = k.min(self.n);
        for i in 0..=upper {
            sum += self.pmf(i);
        }
        sum.min(1.0)
    }

    pub fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    pub fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

impl fmt::Display for Binomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Binomial(n={}, p={})", self.n, self.p)
    }
}

/// Geometric distribution (number of trials until first success).
pub struct Geometric {
    pub p: f64,
}

impl Geometric {
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p <= 1.0);
        Self { p }
    }

    /// PMF: P(X=k) = (1-p)^{k-1} * p for k >= 1.
    pub fn pmf(&self, k: u64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        (1.0 - self.p).powi((k - 1) as i32) * self.p
    }

    /// CDF: P(X <= k) = 1 - (1-p)^k.
    pub fn cdf(&self, k: u64) -> f64 {
        1.0 - (1.0 - self.p).powi(k as i32)
    }

    pub fn quantile(&self, prob: f64) -> u64 {
        assert!(prob >= 0.0 && prob < 1.0);
        if prob <= 0.0 {
            return 1;
        }
        ((1.0 - prob).ln() / (1.0 - self.p).ln()).ceil() as u64
    }
}

impl fmt::Display for Geometric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Geometric(p={})", self.p)
    }
}

/// Beta distribution.
pub struct Beta {
    pub alpha: f64,
    pub beta_param: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta_param: f64) -> Self {
        assert!(alpha > 0.0 && beta_param > 0.0);
        Self { alpha, beta_param }
    }

    /// PDF using the beta function.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        let log_pdf = (self.alpha - 1.0) * x.ln()
            + (self.beta_param - 1.0) * (1.0 - x).ln()
            - ln_beta(self.alpha, self.beta_param);
        log_pdf.exp()
    }

    /// CDF using numerical integration (Simpson's rule).
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        // Numerical integration
        let n = 200;
        let h = x / n as f64;
        let mut sum = self.pdf(0.0) + self.pdf(x);
        for i in 1..n {
            let xi = i as f64 * h;
            if i % 2 == 0 {
                sum += 2.0 * self.pdf(xi);
            } else {
                sum += 4.0 * self.pdf(xi);
            }
        }
        (sum * h / 3.0).clamp(0.0, 1.0)
    }

    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta_param)
    }

    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta_param;
        self.alpha * self.beta_param / (ab * ab * (ab + 1.0))
    }
}

impl fmt::Display for Beta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Beta(alpha={}, beta={})", self.alpha, self.beta_param)
    }
}

/// Chi-squared distribution.
pub struct ChiSquared {
    pub k: u32,
}

impl ChiSquared {
    pub fn new(k: u32) -> Self {
        assert!(k > 0);
        Self { k }
    }

    /// PDF.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let half_k = self.k as f64 / 2.0;
        let log_pdf = (half_k - 1.0) * x.ln() - x / 2.0 - half_k * 2.0_f64.ln() - ln_gamma(half_k);
        log_pdf.exp()
    }

    /// CDF using numerical integration.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // Use Simpson's rule
        let n = 400;
        let h = x / n as f64;
        let mut sum = self.pdf(0.0) + self.pdf(x);
        for i in 1..n {
            let xi = i as f64 * h;
            if i % 2 == 0 {
                sum += 2.0 * self.pdf(xi);
            } else {
                sum += 4.0 * self.pdf(xi);
            }
        }
        (sum * h / 3.0).clamp(0.0, 1.0)
    }

    pub fn mean(&self) -> f64 {
        self.k as f64
    }

    pub fn variance(&self) -> f64 {
        2.0 * self.k as f64
    }
}

impl fmt::Display for ChiSquared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChiSquared(k={})", self.k)
    }
}

// ── Helper functions ─────────────────────────────────────────────

/// Error function approximation (Abramowitz & Stegun 7.1.26).
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Log-gamma function using Stirling's approximation (Lanczos).
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation, g=7
    let coefs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        let reflect =
            std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - ln_gamma(1.0 - x);
        return reflect;
    }

    let x = x - 1.0;
    let mut sum = coefs[0];
    for i in 1..9 {
        sum += coefs[i] / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Log of the beta function: ln(B(a,b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b)).
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Log-factorial using ln_gamma.
fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

/// Log of binomial coefficient C(n, k).
fn ln_binomial_coeff(n: u64, k: u64) -> f64 {
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// Standard normal quantile (inverse CDF) using rational approximation.
fn standard_normal_quantile(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_rng_uniform_range() {
        let mut rng = Rng::new(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_uniform_distribution() {
        let mut rng = Rng::new(42);
        let samples = rng.sample_uniform(0.0, 1.0, 10000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(approx_eq(mean, 0.5, 0.05));
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = Rng::new(42);
        let samples = rng.sample_normal(0.0, 1.0, 10000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let var: f64 = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
            / samples.len() as f64;
        assert!(approx_eq(mean, 0.0, 0.1));
        assert!(approx_eq(var, 1.0, 0.15));
    }

    #[test]
    fn test_exponential_distribution() {
        let mut rng = Rng::new(42);
        let lambda = 2.0;
        let samples = rng.sample_exponential(lambda, 10000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        // E[X] = 1/lambda
        assert!(approx_eq(mean, 1.0 / lambda, 0.05));
        // All values should be positive
        assert!(samples.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn test_poisson_sampling() {
        let mut rng = Rng::new(42);
        let lambda = 5.0;
        let n = 10000;
        let mut sum = 0u64;
        for _ in 0..n {
            sum += rng.poisson(lambda);
        }
        let mean = sum as f64 / n as f64;
        assert!(approx_eq(mean, lambda, 0.3));
    }

    #[test]
    fn test_binomial_sampling() {
        let mut rng = Rng::new(42);
        let n_trials = 10u64;
        let p = 0.3;
        let n_samples = 5000;
        let mut sum = 0u64;
        for _ in 0..n_samples {
            let k = rng.binomial(n_trials, p);
            assert!(k <= n_trials);
            sum += k;
        }
        let mean = sum as f64 / n_samples as f64;
        assert!(approx_eq(mean, n_trials as f64 * p, 0.3));
    }

    #[test]
    fn test_geometric_sampling() {
        let mut rng = Rng::new(42);
        let p = 0.5;
        let n = 5000;
        let mut sum = 0u64;
        for _ in 0..n {
            let k = rng.geometric(p);
            assert!(k >= 1);
            sum += k;
        }
        let mean = sum as f64 / n as f64;
        // E[X] = 1/p
        assert!(approx_eq(mean, 1.0 / p, 0.3));
    }

    #[test]
    fn test_beta_sampling() {
        let mut rng = Rng::new(42);
        let alpha = 2.0;
        let beta_p = 5.0;
        let n = 5000;
        let mut sum = 0.0;
        for _ in 0..n {
            let v = rng.beta(alpha, beta_p);
            assert!(v >= 0.0 && v <= 1.0);
            sum += v;
        }
        let mean = sum / n as f64;
        // E[X] = alpha / (alpha + beta)
        assert!(approx_eq(mean, alpha / (alpha + beta_p), 0.05));
    }

    #[test]
    fn test_chi_squared_sampling() {
        let mut rng = Rng::new(42);
        let k = 4;
        let n = 5000;
        let mut sum = 0.0;
        for _ in 0..n {
            let v = rng.chi_squared(k);
            assert!(v >= 0.0);
            sum += v;
        }
        let mean = sum / n as f64;
        // E[X] = k
        assert!(approx_eq(mean, k as f64, 0.5));
    }

    #[test]
    fn test_uniform_pdf_cdf() {
        let u = Uniform::new(0.0, 1.0);
        assert!(approx_eq(u.pdf(0.5), 1.0, 1e-12));
        assert!(approx_eq(u.pdf(-0.1), 0.0, 1e-12));
        assert!(approx_eq(u.cdf(0.5), 0.5, 1e-12));
        assert!(approx_eq(u.cdf(0.0), 0.0, 1e-12));
        assert!(approx_eq(u.cdf(1.0), 1.0, 1e-12));
    }

    #[test]
    fn test_uniform_quantile() {
        let u = Uniform::new(0.0, 10.0);
        assert!(approx_eq(u.quantile(0.5), 5.0, 1e-12));
        assert!(approx_eq(u.quantile(0.0), 0.0, 1e-12));
    }

    #[test]
    fn test_normal_pdf_cdf() {
        let n = Normal::standard();
        // PDF at 0 should be 1/sqrt(2*pi)
        assert!(approx_eq(n.pdf(0.0), 1.0 / (2.0 * std::f64::consts::PI).sqrt(), 1e-10));
        // CDF at 0 should be 0.5
        assert!(approx_eq(n.cdf(0.0), 0.5, 1e-6));
        // CDF is monotonic
        assert!(n.cdf(1.0) > n.cdf(0.0));
    }

    #[test]
    fn test_normal_quantile() {
        let n = Normal::standard();
        assert!(approx_eq(n.quantile(0.5), 0.0, 1e-6));
        // quantile(CDF(x)) ~ x
        let x = 1.5;
        let p = n.cdf(x);
        assert!(approx_eq(n.quantile(p), x, 0.01));
    }

    #[test]
    fn test_exponential_pdf_cdf() {
        let e = Exponential::new(1.0);
        assert!(approx_eq(e.pdf(0.0), 1.0, 1e-12));
        assert!(approx_eq(e.cdf(0.0), 0.0, 1e-12));
        assert!(approx_eq(e.cdf(1.0), 1.0 - (-1.0_f64).exp(), 1e-10));
    }

    #[test]
    fn test_exponential_quantile() {
        let e = Exponential::new(2.0);
        assert!(approx_eq(e.quantile(0.0), 0.0, 1e-12));
        let x = 0.5;
        let p = e.cdf(x);
        assert!(approx_eq(e.quantile(p), x, 1e-10));
    }

    #[test]
    fn test_poisson_pmf() {
        let p = Poisson::new(3.0);
        // P(X=0) = e^{-3}
        assert!(approx_eq(p.pmf(0), (-3.0_f64).exp(), 1e-10));
        // Sum of PMF over many values should be ~1
        let sum: f64 = (0..30).map(|k| p.pmf(k)).sum();
        assert!(approx_eq(sum, 1.0, 1e-6));
    }

    #[test]
    fn test_binomial_pmf() {
        let b = Binomial::new(10, 0.5);
        // P(X=5) should be C(10,5) * 0.5^10 = 252/1024
        assert!(approx_eq(b.pmf(5), 252.0 / 1024.0, 1e-8));
        assert!(approx_eq(b.mean(), 5.0, 1e-12));
    }

    #[test]
    fn test_geometric_pmf_cdf() {
        let g = Geometric::new(0.5);
        assert!(approx_eq(g.pmf(1), 0.5, 1e-12));
        assert!(approx_eq(g.pmf(2), 0.25, 1e-12));
        assert!(approx_eq(g.cdf(1), 0.5, 1e-12));
    }

    #[test]
    fn test_beta_pdf_mean() {
        let b = Beta::new(2.0, 2.0);
        // Symmetric beta: mean = 0.5
        assert!(approx_eq(b.mean(), 0.5, 1e-12));
        // PDF should be 0 outside [0,1]
        assert!(approx_eq(b.pdf(-0.1), 0.0, 1e-12));
        assert!(approx_eq(b.pdf(1.1), 0.0, 1e-12));
    }

    #[test]
    fn test_chi_squared_pdf() {
        let cs = ChiSquared::new(2);
        // Chi-sq(2) is Exponential(0.5): PDF at 0 = 0.5
        assert!(approx_eq(cs.pdf(0.01), 0.5 * (-0.005_f64).exp(), 0.05));
        assert!(approx_eq(cs.mean(), 2.0, 1e-12));
        assert!(approx_eq(cs.variance(), 4.0, 1e-12));
    }

    #[test]
    fn test_erf() {
        assert!(approx_eq(erf(0.0), 0.0, 1e-12));
        assert!(approx_eq(erf(100.0), 1.0, 1e-6));
        assert!(approx_eq(erf(-100.0), -1.0, 1e-6));
    }

    #[test]
    fn test_distribution_display() {
        assert_eq!(format!("{}", Uniform::new(0.0, 1.0)), "Uniform(0, 1)");
        assert!(format!("{}", Normal::standard()).contains("Normal"));
        assert!(format!("{}", Exponential::new(1.0)).contains("Exponential"));
    }
}
