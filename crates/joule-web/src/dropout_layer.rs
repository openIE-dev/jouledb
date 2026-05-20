//! Dropout regularization layers for neural networks.
//!
//! Implements standard dropout, inverted dropout (the modern default),
//! spatial/channel dropout for convolutional layers, and alpha dropout
//! for self-normalizing networks (SELU).

use std::fmt;

// ── Deterministic PRNG ────────────────────────────────────────────

/// Xoshiro256** PRNG for fast, reproducible dropout masks.
#[derive(Debug, Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        // SplitMix64 to seed the state
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ── DropoutMode ───────────────────────────────────────────────────

/// Dropout variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DropoutVariant {
    /// Standard dropout: zero out, no scaling during training; scale at test time.
    Standard,
    /// Inverted dropout: zero out and scale up by 1/(1-p) during training.
    Inverted,
}

impl fmt::Display for DropoutVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::Inverted => write!(f, "inverted"),
        }
    }
}

// ── DropoutLayer ──────────────────────────────────────────────────

/// Dropout layer with configurable drop rate and variant.
#[derive(Debug, Clone)]
pub struct DropoutLayer {
    pub drop_rate: f64,
    pub variant: DropoutVariant,
    pub training: bool,
    rng: Xoshiro256,
    last_mask: Vec<bool>,
}

impl DropoutLayer {
    /// Create a new dropout layer.
    ///
    /// `drop_rate` is the probability of dropping a unit (0.0 to 1.0).
    pub fn new(drop_rate: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&drop_rate),
            "drop rate must be in [0, 1]"
        );
        Self {
            drop_rate,
            variant: DropoutVariant::Inverted,
            training: true,
            rng: Xoshiro256::new(42),
            last_mask: Vec::new(),
        }
    }

    pub fn with_variant(mut self, variant: DropoutVariant) -> Self {
        self.variant = variant;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Xoshiro256::new(seed);
        self
    }

    pub fn with_training(mut self, training: bool) -> Self {
        self.training = training;
        self
    }

    /// Set training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation (inference) mode.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Generate a dropout mask: `true` means *keep*, `false` means *drop*.
    fn generate_mask(&mut self, size: usize) -> Vec<bool> {
        (0..size)
            .map(|_| self.rng.next_f64() >= self.drop_rate)
            .collect()
    }

    /// Forward pass.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        if !self.training || self.drop_rate == 0.0 {
            return input.to_vec();
        }

        if self.drop_rate >= 1.0 {
            self.last_mask = vec![false; input.len()];
            return vec![0.0; input.len()];
        }

        let mask = self.generate_mask(input.len());
        let scale = match self.variant {
            DropoutVariant::Inverted => 1.0 / (1.0 - self.drop_rate),
            DropoutVariant::Standard => 1.0,
        };

        let output: Vec<f64> = input
            .iter()
            .zip(mask.iter())
            .map(|(x, keep)| if *keep { x * scale } else { 0.0 })
            .collect();

        self.last_mask = mask;
        output
    }

    /// Backward pass: gradient flows only through kept units.
    pub fn backward(&self, grad_output: &[f64]) -> Vec<f64> {
        if !self.training || self.drop_rate == 0.0 {
            return grad_output.to_vec();
        }

        let scale = match self.variant {
            DropoutVariant::Inverted => 1.0 / (1.0 - self.drop_rate),
            DropoutVariant::Standard => 1.0,
        };

        grad_output
            .iter()
            .zip(self.last_mask.iter())
            .map(|(g, keep)| if *keep { g * scale } else { 0.0 })
            .collect()
    }

    /// Fraction of units actually dropped in the last forward pass.
    pub fn actual_drop_fraction(&self) -> f64 {
        if self.last_mask.is_empty() {
            return 0.0;
        }
        let dropped = self.last_mask.iter().filter(|&&k| !k).count();
        dropped as f64 / self.last_mask.len() as f64
    }

    /// Get the last mask (for inspection or debugging).
    pub fn last_mask(&self) -> &[bool] {
        &self.last_mask
    }
}

impl fmt::Display for DropoutLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dropout(p={:.2}, variant={}, training={})",
            self.drop_rate, self.variant, self.training
        )
    }
}

// ── Spatial Dropout ───────────────────────────────────────────────

/// Spatial (channel-wise) dropout for convolutional feature maps.
///
/// Drops entire channels rather than individual elements,
/// preserving spatial structure within kept channels.
#[derive(Debug, Clone)]
pub struct SpatialDropout {
    pub drop_rate: f64,
    pub training: bool,
    rng: Xoshiro256,
    channel_mask: Vec<bool>,
}

impl SpatialDropout {
    pub fn new(drop_rate: f64) -> Self {
        assert!((0.0..=1.0).contains(&drop_rate));
        Self {
            drop_rate,
            training: true,
            rng: Xoshiro256::new(123),
            channel_mask: Vec::new(),
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Xoshiro256::new(seed);
        self
    }

    /// Forward pass. Input: `[channels, height, width]` flattened.
    pub fn forward(
        &mut self,
        input: &[f64],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<f64> {
        let spatial = height * width;
        assert_eq!(input.len(), channels * spatial);

        if !self.training || self.drop_rate == 0.0 {
            return input.to_vec();
        }

        // Generate one mask value per channel
        self.channel_mask = (0..channels)
            .map(|_| self.rng.next_f64() >= self.drop_rate)
            .collect();

        let scale = 1.0 / (1.0 - self.drop_rate);
        let mut output = vec![0.0; input.len()];

        for c in 0..channels {
            if self.channel_mask[c] {
                for s in 0..spatial {
                    output[c * spatial + s] = input[c * spatial + s] * scale;
                }
            }
        }

        output
    }

    /// Get the channel keep mask from the last forward pass.
    pub fn channel_mask(&self) -> &[bool] {
        &self.channel_mask
    }

    /// Count of kept channels in last forward.
    pub fn kept_channels(&self) -> usize {
        self.channel_mask.iter().filter(|&&k| k).count()
    }
}

impl fmt::Display for SpatialDropout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialDropout(p={:.2})", self.drop_rate)
    }
}

// ── Alpha Dropout ─────────────────────────────────────────────────

/// Alpha dropout for self-normalizing networks (used with SELU).
///
/// Instead of setting dropped units to zero, sets them to the
/// negative saturation value of SELU to maintain self-normalizing
/// properties.
#[derive(Debug, Clone)]
pub struct AlphaDropout {
    pub drop_rate: f64,
    pub training: bool,
    rng: Xoshiro256,
}

/// SELU constants.
const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_LAMBDA: f64 = 1.0507009873554805;

impl AlphaDropout {
    pub fn new(drop_rate: f64) -> Self {
        assert!((0.0..=1.0).contains(&drop_rate));
        Self {
            drop_rate,
            training: true,
            rng: Xoshiro256::new(777),
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Xoshiro256::new(seed);
        self
    }

    /// Forward pass with alpha-dropout transformation.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        if !self.training || self.drop_rate == 0.0 {
            return input.to_vec();
        }

        let sat = -SELU_LAMBDA * SELU_ALPHA;
        let p = self.drop_rate;
        let q = 1.0 - p;

        // Affine correction to maintain mean and variance
        let a = (q + p * q * sat * sat).sqrt().recip();
        let b_val = -a * p * sat;

        input
            .iter()
            .map(|x| {
                if self.rng.next_f64() >= self.drop_rate {
                    a * x + b_val
                } else {
                    a * sat + b_val
                }
            })
            .collect()
    }
}

impl fmt::Display for AlphaDropout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlphaDropout(p={:.2})", self.drop_rate)
    }
}

// ── DropoutSchedule ───────────────────────────────────────────────

/// Linearly schedule dropout rate across training.
#[derive(Debug, Clone)]
pub struct DropoutSchedule {
    pub start_rate: f64,
    pub end_rate: f64,
    pub total_steps: usize,
}

impl DropoutSchedule {
    pub fn new(start_rate: f64, end_rate: f64, total_steps: usize) -> Self {
        Self { start_rate, end_rate, total_steps }
    }

    /// Get the dropout rate at a given training step.
    pub fn rate_at(&self, step: usize) -> f64 {
        if step >= self.total_steps {
            return self.end_rate;
        }
        let t = step as f64 / self.total_steps as f64;
        self.start_rate + (self.end_rate - self.start_rate) * t
    }
}

impl fmt::Display for DropoutSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DropoutSchedule({:.2} -> {:.2} over {} steps)",
            self.start_rate, self.end_rate, self.total_steps
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_creation() {
        let d = DropoutLayer::new(0.5);
        assert!((d.drop_rate - 0.5).abs() < 1e-10);
        assert!(d.training);
    }

    #[test]
    fn test_dropout_eval_passthrough() {
        let mut d = DropoutLayer::new(0.5).with_training(false);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = d.forward(&input);
        assert_eq!(out, input);
    }

    #[test]
    fn test_dropout_zero_rate() {
        let mut d = DropoutLayer::new(0.0);
        let input = vec![1.0, 2.0, 3.0];
        let out = d.forward(&input);
        assert_eq!(out, input);
    }

    #[test]
    fn test_dropout_full_rate() {
        let mut d = DropoutLayer::new(1.0);
        let input = vec![1.0, 2.0, 3.0];
        let out = d.forward(&input);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_inverted_dropout_scale() {
        let mut d = DropoutLayer::new(0.5).with_seed(42);
        let input = vec![2.0; 1000];
        let out = d.forward(&input);
        // Inverted dropout scales kept values by 1/(1-0.5) = 2.0
        for (&o, &keep) in out.iter().zip(d.last_mask().iter()) {
            if keep {
                assert!((o - 4.0).abs() < 1e-10);
            } else {
                assert!((o).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_standard_dropout_no_scale() {
        let mut d = DropoutLayer::new(0.5)
            .with_variant(DropoutVariant::Standard)
            .with_seed(42);
        let input = vec![3.0; 100];
        let out = d.forward(&input);
        for (&o, &keep) in out.iter().zip(d.last_mask().iter()) {
            if keep {
                assert!((o - 3.0).abs() < 1e-10);
            } else {
                assert!((o).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dropout_actual_fraction() {
        let mut d = DropoutLayer::new(0.5).with_seed(0);
        d.forward(&vec![1.0; 10000]);
        let frac = d.actual_drop_fraction();
        // Should be roughly 0.5 with 10000 samples
        assert!((frac - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_dropout_backward() {
        let mut d = DropoutLayer::new(0.5).with_seed(99);
        let input = vec![1.0; 10];
        d.forward(&input);
        let grad = vec![1.0; 10];
        let grad_in = d.backward(&grad);
        // Gradient should be 0 where mask is false, scaled where true
        for (i, &keep) in d.last_mask().iter().enumerate() {
            if !keep {
                assert!((grad_in[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dropout_backward_eval() {
        let mut d = DropoutLayer::new(0.5).with_training(false);
        let grad = vec![1.0, 2.0, 3.0];
        let grad_in = d.backward(&grad);
        assert_eq!(grad_in, grad);
    }

    #[test]
    fn test_spatial_dropout_creation() {
        let sd = SpatialDropout::new(0.3);
        assert!((sd.drop_rate - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_dropout_forward() {
        let mut sd = SpatialDropout::new(0.5).with_seed(42);
        let input = vec![1.0; 4 * 3 * 3]; // 4 channels, 3x3
        let out = sd.forward(&input, 4, 3, 3);
        assert_eq!(out.len(), 36);
        // Entire channels should be either all zero or all scaled
        for c in 0..4 {
            let chan: Vec<f64> = (0..9).map(|s| out[c * 9 + s]).collect();
            let all_zero = chan.iter().all(|v| *v == 0.0);
            let all_scaled = chan.iter().all(|v| (v - 2.0).abs() < 1e-10);
            assert!(all_zero || all_scaled);
        }
    }

    #[test]
    fn test_spatial_dropout_eval() {
        let mut sd = SpatialDropout::new(0.5);
        sd.training = false;
        let input = vec![1.0; 2 * 4 * 4];
        let out = sd.forward(&input, 2, 4, 4);
        assert_eq!(out, input);
    }

    #[test]
    fn test_spatial_dropout_kept_channels() {
        let mut sd = SpatialDropout::new(0.5).with_seed(0);
        sd.forward(&vec![1.0; 100 * 4 * 4], 100, 4, 4);
        let kept = sd.kept_channels();
        assert!(kept > 0 && kept < 100);
    }

    #[test]
    fn test_alpha_dropout() {
        let mut ad = AlphaDropout::new(0.1).with_seed(42);
        let input = vec![1.0; 100];
        let out = ad.forward(&input);
        assert_eq!(out.len(), 100);
        // Dropped values should NOT be zero (alpha dropout sets to saturation)
        let zeros = out.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros < 50); // Most values won't be exactly zero
    }

    #[test]
    fn test_alpha_dropout_eval() {
        let mut ad = AlphaDropout::new(0.5);
        ad.training = false;
        let input = vec![1.0, 2.0, 3.0];
        let out = ad.forward(&input);
        assert_eq!(out, input);
    }

    #[test]
    fn test_dropout_schedule() {
        let sched = DropoutSchedule::new(0.0, 0.5, 100);
        assert!((sched.rate_at(0) - 0.0).abs() < 1e-10);
        assert!((sched.rate_at(50) - 0.25).abs() < 1e-10);
        assert!((sched.rate_at(100) - 0.5).abs() < 1e-10);
        assert!((sched.rate_at(200) - 0.5).abs() < 1e-10); // clamped
    }

    #[test]
    fn test_dropout_display() {
        let d = DropoutLayer::new(0.3);
        let s = format!("{}", d);
        assert!(s.contains("0.30"));
        assert!(s.contains("inverted"));
    }

    #[test]
    fn test_spatial_dropout_display() {
        let sd = SpatialDropout::new(0.2);
        assert_eq!(format!("{}", sd), "SpatialDropout(p=0.20)");
    }

    #[test]
    fn test_alpha_dropout_display() {
        let ad = AlphaDropout::new(0.1);
        assert_eq!(format!("{}", ad), "AlphaDropout(p=0.10)");
    }

    #[test]
    fn test_schedule_display() {
        let sched = DropoutSchedule::new(0.0, 0.5, 1000);
        let s = format!("{}", sched);
        assert!(s.contains("0.00"));
        assert!(s.contains("0.50"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn test_variant_display() {
        assert_eq!(format!("{}", DropoutVariant::Standard), "standard");
        assert_eq!(format!("{}", DropoutVariant::Inverted), "inverted");
    }
}
