//! Weight initialization strategies for neural network layers.
//!
//! Proper initialization is critical for stable training — these methods
//! set initial parameter values to maintain variance through forward and
//! backward passes:
//!
//! - [`XavierInit`] — Glorot uniform/normal for tanh/sigmoid activations
//! - [`HeInit`] — Kaiming uniform/normal for ReLU activations
//! - [`OrthogonalInit`] — orthogonal matrices via Gram-Schmidt
//! - [`UniformInit`] — bounded uniform random initialization
//! - [`NormalInit`] — Gaussian random initialization
//! - [`SparseInit`] — sparse initialization with controlled density
//! - [`ConstantInit`] — fill with a constant value (e.g., zeros for biases)

use std::fmt;

// ── PRNG ───────────────────────────────────────────────────────────

/// Simple xoshiro256** PRNG for deterministic initialization.
/// No external crate dependencies.
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed into 4 state words
        let mut s = seed;
        let mut state = [0u64; 4];
        for slot in &mut state {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [low, high).
    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + self.next_f64() * (high - low)
    }

    /// Standard normal via Box-Muller transform.
    fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    }
}

// ── Distribution Mode ──────────────────────────────────────────────

/// Whether to sample from a uniform or normal distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistMode {
    Uniform,
    Normal,
}

// ── Xavier / Glorot ────────────────────────────────────────────────

/// Xavier/Glorot initialization (Glorot & Bengio, 2010).
///
/// Designed for layers with tanh or sigmoid activations. Maintains
/// variance by scaling based on both fan-in and fan-out:
///
/// - **Uniform**: `U(-limit, limit)` where `limit = √(6 / (fan_in + fan_out))`
/// - **Normal**: `N(0, σ)` where `σ = √(2 / (fan_in + fan_out))`
pub struct XavierInit {
    mode: DistMode,
    seed: u64,
    gain: f64,
}

impl XavierInit {
    pub fn new() -> Self {
        Self {
            mode: DistMode::Uniform,
            seed: 42,
            gain: 1.0,
        }
    }

    pub fn with_mode(mut self, mode: DistMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }

    /// Initialize a weight matrix for a layer with `fan_in` inputs and `fan_out` outputs.
    pub fn initialize(&self, fan_in: usize, fan_out: usize) -> Vec<f64> {
        let count = fan_in * fan_out;
        let mut rng = Rng::new(self.seed);
        let mut weights = Vec::with_capacity(count);

        match self.mode {
            DistMode::Uniform => {
                let limit = self.gain * (6.0 / (fan_in + fan_out) as f64).sqrt();
                for _ in 0..count {
                    weights.push(rng.uniform(-limit, limit));
                }
            }
            DistMode::Normal => {
                let std_dev = self.gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
                for _ in 0..count {
                    weights.push(rng.normal(0.0, std_dev));
                }
            }
        }

        weights
    }
}

impl Default for XavierInit {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for XavierInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "XavierInit(mode={:?}, gain={:.4})", self.mode, self.gain)
    }
}

// ── He / Kaiming ───────────────────────────────────────────────────

/// He/Kaiming initialization (He et al., 2015).
///
/// Designed for layers with ReLU activations. Accounts for the fact
/// that ReLU zeros out half the values:
///
/// - **Uniform**: `U(-limit, limit)` where `limit = √(6 / fan_in)`  (mode='fan_in')
/// - **Normal**: `N(0, σ)` where `σ = √(2 / fan_in)`
pub struct HeInit {
    mode: DistMode,
    fan_mode: FanMode,
    seed: u64,
    negative_slope: f64,
}

/// Whether to use fan-in or fan-out for variance computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FanMode {
    FanIn,
    FanOut,
}

impl HeInit {
    pub fn new() -> Self {
        Self {
            mode: DistMode::Normal,
            fan_mode: FanMode::FanIn,
            seed: 42,
            negative_slope: 0.0,
        }
    }

    pub fn with_mode(mut self, mode: DistMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_fan_mode(mut self, fan_mode: FanMode) -> Self {
        self.fan_mode = fan_mode;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// For leaky ReLU with a nonzero negative slope.
    pub fn with_negative_slope(mut self, slope: f64) -> Self {
        self.negative_slope = slope;
        self
    }

    pub fn initialize(&self, fan_in: usize, fan_out: usize) -> Vec<f64> {
        let count = fan_in * fan_out;
        let fan = match self.fan_mode {
            FanMode::FanIn => fan_in,
            FanMode::FanOut => fan_out,
        };
        let gain = (2.0 / (1.0 + self.negative_slope * self.negative_slope)).sqrt();
        let mut rng = Rng::new(self.seed);
        let mut weights = Vec::with_capacity(count);

        match self.mode {
            DistMode::Uniform => {
                let limit = gain * (3.0 / fan as f64).sqrt();
                for _ in 0..count {
                    weights.push(rng.uniform(-limit, limit));
                }
            }
            DistMode::Normal => {
                let std_dev = gain / (fan as f64).sqrt();
                for _ in 0..count {
                    weights.push(rng.normal(0.0, std_dev));
                }
            }
        }

        weights
    }
}

impl Default for HeInit {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for HeInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HeInit(mode={:?}, fan={:?}, slope={:.4})",
            self.mode, self.fan_mode, self.negative_slope
        )
    }
}

// ── Orthogonal ─────────────────────────────────────────────────────

/// Orthogonal initialization via modified Gram-Schmidt.
///
/// Generates a matrix whose columns (or rows) are orthonormal.
/// Particularly effective for RNNs where it prevents vanishing/exploding
/// gradients through long sequences.
pub struct OrthogonalInit {
    gain: f64,
    seed: u64,
}

impl OrthogonalInit {
    pub fn new() -> Self {
        Self { gain: 1.0, seed: 42 }
    }

    pub fn with_gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns `rows * cols` values forming a (partially) orthogonal matrix.
    pub fn initialize(&self, rows: usize, cols: usize) -> Vec<f64> {
        let mut rng = Rng::new(self.seed);
        let n = rows.max(cols);

        // Generate random matrix
        let mut mat = vec![0.0; n * n];
        for val in mat.iter_mut() {
            *val = rng.normal(0.0, 1.0);
        }

        // Modified Gram-Schmidt orthogonalization
        for i in 0..n {
            // Normalize column i
            let norm = col_norm(&mat, n, i);
            if norm > 1e-15 {
                for r in 0..n {
                    mat[r * n + i] /= norm;
                }
            }
            // Subtract projections from subsequent columns
            for j in (i + 1)..n {
                let dot = col_dot(&mat, n, i, j);
                for r in 0..n {
                    mat[r * n + j] -= dot * mat[r * n + i];
                }
            }
        }

        // Extract the requested submatrix and apply gain
        let mut result = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                result.push(self.gain * mat[r * n + c]);
            }
        }
        result
    }
}

impl Default for OrthogonalInit {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for OrthogonalInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrthogonalInit(gain={:.4})", self.gain)
    }
}

/// Dot product of columns i and j in an n×n matrix.
fn col_dot(mat: &[f64], n: usize, i: usize, j: usize) -> f64 {
    let mut dot = 0.0;
    for r in 0..n {
        dot += mat[r * n + i] * mat[r * n + j];
    }
    dot
}

/// L2 norm of column i in an n×n matrix.
fn col_norm(mat: &[f64], n: usize, i: usize) -> f64 {
    col_dot(mat, n, i, i).sqrt()
}

// ── Uniform ────────────────────────────────────────────────────────

/// Simple bounded uniform initialization: `U(low, high)`.
pub struct UniformInit {
    low: f64,
    high: f64,
    seed: u64,
}

impl UniformInit {
    pub fn new(low: f64, high: f64) -> Self {
        Self { low, high, seed: 42 }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_range(mut self, low: f64, high: f64) -> Self {
        self.low = low;
        self.high = high;
        self
    }

    pub fn initialize(&self, count: usize) -> Vec<f64> {
        let mut rng = Rng::new(self.seed);
        (0..count).map(|_| rng.uniform(self.low, self.high)).collect()
    }
}

impl fmt::Display for UniformInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UniformInit(U[{:.4}, {:.4}])", self.low, self.high)
    }
}

// ── Normal ─────────────────────────────────────────────────────────

/// Gaussian initialization: `N(mean, std)`.
pub struct NormalInit {
    mean: f64,
    std_dev: f64,
    seed: u64,
}

impl NormalInit {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self {
            mean,
            std_dev: std_dev.abs(),
            seed: 42,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_std(mut self, std_dev: f64) -> Self {
        self.std_dev = std_dev.abs();
        self
    }

    pub fn initialize(&self, count: usize) -> Vec<f64> {
        let mut rng = Rng::new(self.seed);
        (0..count).map(|_| rng.normal(self.mean, self.std_dev)).collect()
    }
}

impl fmt::Display for NormalInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NormalInit(N({:.4}, {:.4}))", self.mean, self.std_dev)
    }
}

// ── Sparse ─────────────────────────────────────────────────────────

/// Sparse initialization: only a fraction of weights are nonzero,
/// drawn from `N(0, std)`. The rest are zero.
///
/// Useful for very large layers where full dense initialization
/// would lead to excessive computation in early training.
pub struct SparseInit {
    sparsity: f64,
    std_dev: f64,
    seed: u64,
}

impl SparseInit {
    pub fn new(sparsity: f64) -> Self {
        Self {
            sparsity: sparsity.clamp(0.0, 1.0),
            std_dev: 0.01,
            seed: 42,
        }
    }

    pub fn with_std(mut self, std_dev: f64) -> Self {
        self.std_dev = std_dev.abs();
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_sparsity(mut self, sparsity: f64) -> Self {
        self.sparsity = sparsity.clamp(0.0, 1.0);
        self
    }

    pub fn initialize(&self, count: usize) -> Vec<f64> {
        let mut rng = Rng::new(self.seed);
        (0..count)
            .map(|_| {
                if rng.next_f64() < self.sparsity {
                    0.0
                } else {
                    rng.normal(0.0, self.std_dev)
                }
            })
            .collect()
    }

    /// Fraction of nonzero weights in a generated vector.
    pub fn expected_density(&self) -> f64 {
        1.0 - self.sparsity
    }
}

impl fmt::Display for SparseInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseInit(sparsity={:.2}%, σ={:.4})",
            self.sparsity * 100.0,
            self.std_dev
        )
    }
}

// ── Constant ───────────────────────────────────────────────────────

/// Fill all weights with a constant value (zeros, ones, or custom).
pub struct ConstantInit {
    value: f64,
}

impl ConstantInit {
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    pub fn zeros() -> Self {
        Self { value: 0.0 }
    }

    pub fn ones() -> Self {
        Self { value: 1.0 }
    }

    pub fn with_value(mut self, value: f64) -> Self {
        self.value = value;
        self
    }

    pub fn initialize(&self, count: usize) -> Vec<f64> {
        vec![self.value; count]
    }
}

impl fmt::Display for ConstantInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConstantInit(value={:.6})", self.value)
    }
}

// ── Initialization Statistics ──────────────────────────────────────

/// Compute summary statistics to verify initialization quality.
#[derive(Debug, Clone)]
pub struct InitStats {
    pub mean: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub sparsity: f64,
    pub count: usize,
}

impl InitStats {
    pub fn compute(weights: &[f64]) -> Self {
        if weights.is_empty() {
            return Self {
                mean: 0.0,
                variance: 0.0,
                min: 0.0,
                max: 0.0,
                sparsity: 1.0,
                count: 0,
            };
        }
        let n = weights.len() as f64;
        let sum: f64 = weights.iter().sum();
        let mean = sum / n;
        let variance = weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
        let min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let zeros = weights.iter().filter(|w| w.abs() < 1e-15).count();
        Self {
            mean,
            variance,
            min,
            max,
            sparsity: zeros as f64 / n,
            count: weights.len(),
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }
}

impl fmt::Display for InitStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InitStats(μ={:.4e}, σ={:.4e}, range=[{:.4}, {:.4}], n={})",
            self.mean,
            self.std_dev(),
            self.min,
            self.max,
            self.count
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xavier_uniform_bounds() {
        let weights = XavierInit::new()
            .with_mode(DistMode::Uniform)
            .initialize(100, 100);
        let limit = (6.0 / 200.0_f64).sqrt();
        for &w in &weights {
            assert!(w >= -limit && w <= limit);
        }
    }

    #[test]
    fn xavier_normal_mean_near_zero() {
        let weights = XavierInit::new()
            .with_mode(DistMode::Normal)
            .with_seed(123)
            .initialize(200, 200);
        let stats = InitStats::compute(&weights);
        assert!(stats.mean.abs() < 0.01);
    }

    #[test]
    fn xavier_variance_scales() {
        let small = InitStats::compute(&XavierInit::new().initialize(10, 10));
        let large = InitStats::compute(&XavierInit::new().initialize(1000, 1000));
        // Larger layers should have smaller variance
        assert!(large.variance < small.variance);
    }

    #[test]
    fn xavier_gain() {
        let w1 = XavierInit::new().with_gain(1.0).initialize(50, 50);
        let w2 = XavierInit::new().with_gain(2.0).initialize(50, 50);
        let s1 = InitStats::compute(&w1);
        let s2 = InitStats::compute(&w2);
        // Doubled gain should roughly quadruple variance
        assert!(s2.variance > s1.variance * 3.0);
    }

    #[test]
    fn xavier_display() {
        let init = XavierInit::new();
        assert!(format!("{init}").contains("Xavier"));
    }

    #[test]
    fn he_normal_variance() {
        let weights = HeInit::new()
            .with_mode(DistMode::Normal)
            .with_seed(99)
            .initialize(500, 500);
        let stats = InitStats::compute(&weights);
        // Expected variance ≈ 2/fan_in = 2/500 = 0.004
        assert!((stats.variance - 0.004).abs() < 0.002);
    }

    #[test]
    fn he_fan_out_mode() {
        let fan_in = HeInit::new().with_fan_mode(FanMode::FanIn).initialize(100, 200);
        let fan_out = HeInit::new().with_fan_mode(FanMode::FanOut).initialize(100, 200);
        let s_in = InitStats::compute(&fan_in);
        let s_out = InitStats::compute(&fan_out);
        // Fan-in (100) gives larger variance than fan-out (200)
        assert!(s_in.variance > s_out.variance);
    }

    #[test]
    fn he_leaky_relu() {
        let standard = HeInit::new().initialize(100, 100);
        let leaky = HeInit::new().with_negative_slope(0.2).initialize(100, 100);
        let s1 = InitStats::compute(&standard);
        let s2 = InitStats::compute(&leaky);
        // Leaky ReLU gain is smaller → slightly different variance
        assert!((s1.variance - s2.variance).abs() > 0.0);
    }

    #[test]
    fn he_display() {
        let init = HeInit::new();
        assert!(format!("{init}").contains("HeInit"));
    }

    #[test]
    fn orthogonal_columns_unit_norm() {
        let weights = OrthogonalInit::new().with_seed(7).initialize(4, 4);
        // Check column norms ≈ 1.0
        for c in 0..4 {
            let norm: f64 = (0..4).map(|r| weights[r * 4 + c].powi(2)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 0.1, "column {c} norm = {norm}");
        }
    }

    #[test]
    fn orthogonal_dot_product_near_zero() {
        let weights = OrthogonalInit::new().with_seed(7).initialize(4, 4);
        // Dot product of columns 0 and 1 should be ~0
        let dot: f64 = (0..4).map(|r| weights[r * 4] * weights[r * 4 + 1]).sum();
        assert!(dot.abs() < 0.1);
    }

    #[test]
    fn uniform_in_range() {
        let weights = UniformInit::new(-0.5, 0.5).with_seed(42).initialize(1000);
        for &w in &weights {
            assert!(w >= -0.5 && w < 0.5);
        }
    }

    #[test]
    fn normal_mean_and_std() {
        let weights = NormalInit::new(0.0, 0.1).with_seed(42).initialize(10000);
        let stats = InitStats::compute(&weights);
        assert!(stats.mean.abs() < 0.01);
        assert!((stats.std_dev() - 0.1).abs() < 0.01);
    }

    #[test]
    fn sparse_density() {
        let weights = SparseInit::new(0.8).with_seed(42).initialize(10000);
        let stats = InitStats::compute(&weights);
        // ~80% should be zero
        assert!((stats.sparsity - 0.8).abs() < 0.05);
    }

    #[test]
    fn constant_zeros() {
        let weights = ConstantInit::zeros().initialize(100);
        assert!(weights.iter().all(|w| *w == 0.0));
    }

    #[test]
    fn constant_ones() {
        let weights = ConstantInit::ones().initialize(5);
        assert!(weights.iter().all(|w| (w - 1.0).abs() < 1e-15));
    }

    #[test]
    fn deterministic_seeds() {
        let w1 = XavierInit::new().with_seed(123).initialize(50, 50);
        let w2 = XavierInit::new().with_seed(123).initialize(50, 50);
        assert_eq!(w1, w2);
    }

    #[test]
    fn different_seeds_differ() {
        let w1 = XavierInit::new().with_seed(1).initialize(50, 50);
        let w2 = XavierInit::new().with_seed(2).initialize(50, 50);
        assert_ne!(w1, w2);
    }

    #[test]
    fn init_stats_display() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = InitStats::compute(&weights);
        assert!(format!("{stats}").contains("InitStats"));
    }
}
