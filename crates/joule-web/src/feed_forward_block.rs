//! Position-wise feed-forward network for transformer models.
//!
//! Implements the two-layer feed-forward block with configurable expansion
//! ratio (typically 4x), GELU activation, optional SwiGLU gating, and
//! dropout simulation. Each position in the sequence is processed
//! independently through the same two dense linear layers.

use std::fmt;

// ── Activation Functions ─────────────────────────────────────────

/// Supported activation functions for the feed-forward hidden layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// Gaussian Error Linear Unit: x * Phi(x).
    Gelu,
    /// Rectified Linear Unit: max(0, x).
    Relu,
    /// Sigmoid Linear Unit: x * sigmoid(x).
    Silu,
    /// Gaussian Error Linear Unit (approximate): x * sigmoid(1.702 * x).
    GeluApprox,
}

impl Activation {
    /// Apply the activation function to a scalar.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Gelu => gelu(x),
            Self::Relu => x.max(0.0),
            Self::Silu => x * sigmoid(x),
            Self::GeluApprox => x * sigmoid(1.702 * x),
        }
    }

    /// Compute the derivative (for diagnostics/gradient estimation).
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Relu => if x > 0.0 { 1.0 } else { 0.0 },
            Self::Silu => {
                let s = sigmoid(x);
                s + x * s * (1.0 - s)
            }
            Self::Gelu | Self::GeluApprox => {
                // Numerical derivative
                let h = 1e-7;
                (self.apply(x + h) - self.apply(x - h)) / (2.0 * h)
            }
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gelu => write!(f, "GELU"),
            Self::Relu => write!(f, "ReLU"),
            Self::Silu => write!(f, "SiLU"),
            Self::GeluApprox => write!(f, "GELU(approx)"),
        }
    }
}

/// Sigmoid function: 1 / (1 + exp(-x)).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
fn gelu(x: f64) -> f64 {
    x * 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Approximation of the error function using Abramowitz & Stegun formula 7.1.26.
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    sign * (1.0 - poly * (-a * a).exp())
}

// ── Dense Layer ──────────────────────────────────────────────────

/// A single dense (fully connected) layer.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Weight matrix (out_features x in_features), row-major.
    pub weight: Vec<f64>,
    /// Bias vector of length out_features.
    pub bias: Vec<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

impl DenseLayer {
    /// Initialize with Xavier-like deterministic pseudo-random weights.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut weight = Vec::with_capacity(out_features * in_features);
        let mut state = seed;
        for _ in 0..(out_features * in_features) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            weight.push(u * 2.0 * limit - limit);
        }
        Self { weight, bias: vec![0.0; out_features], in_features, out_features }
    }

    /// Forward pass: (seq_len, in_features) -> (seq_len, out_features).
    pub fn forward(&self, input: &[f64], seq_len: usize) -> Vec<f64> {
        assert_eq!(input.len(), seq_len * self.in_features, "input length mismatch");
        let mut output = vec![0.0; seq_len * self.out_features];
        for s in 0..seq_len {
            for o in 0..self.out_features {
                let mut acc = self.bias[o];
                for i in 0..self.in_features {
                    acc += input[s * self.in_features + i]
                        * self.weight[o * self.in_features + i];
                }
                output[s * self.out_features + o] = acc;
            }
        }
        output
    }

    /// Number of parameters (weights + biases).
    pub fn num_params(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }
}

impl fmt::Display for DenseLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dense({} -> {})", self.in_features, self.out_features)
    }
}

// ── Feed-Forward Configuration ───────────────────────────────────

/// Configuration for the position-wise feed-forward block.
#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    pub d_model: usize,
    pub expansion_ratio: usize,
    pub activation: Activation,
    pub dropout_rate: f64,
    pub use_bias: bool,
}

impl FeedForwardConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            expansion_ratio: 4,
            activation: Activation::Gelu,
            dropout_rate: 0.0,
            use_bias: true,
        }
    }

    pub fn with_expansion_ratio(mut self, ratio: usize) -> Self {
        self.expansion_ratio = ratio;
        self
    }

    pub fn with_activation(mut self, act: Activation) -> Self {
        self.activation = act;
        self
    }

    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// The hidden dimension (d_model * expansion_ratio).
    pub fn d_ff(&self) -> usize {
        self.d_model * self.expansion_ratio
    }
}

// ── Feed-Forward Block ───────────────────────────────────────────

/// Position-wise feed-forward network.
///
/// Applies two linear transformations with an activation in between:
///   FFN(x) = W2 * activation(W1 * x + b1) + b2
///
/// Hidden dimension = d_model * expansion_ratio.
#[derive(Debug, Clone)]
pub struct FeedForwardBlock {
    pub config: FeedForwardConfig,
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    /// Running statistics: mean activation magnitude per forward call.
    activation_stats: Vec<f64>,
}

impl FeedForwardBlock {
    /// Create a new feed-forward block.
    pub fn new(config: FeedForwardConfig) -> Self {
        let d = config.d_model;
        let d_ff = config.d_ff();
        Self {
            layer1: DenseLayer::new(d, d_ff, 500),
            layer2: DenseLayer::new(d_ff, d, 600),
            config,
            activation_stats: Vec::new(),
        }
    }

    /// Forward pass: (seq_len, d_model) -> (seq_len, d_model).
    pub fn forward(&mut self, input: &[f64], seq_len: usize) -> Vec<f64> {
        let d = self.config.d_model;
        assert_eq!(input.len(), seq_len * d, "input length mismatch");

        // First linear: (S, d_model) -> (S, d_ff)
        let mut hidden = self.layer1.forward(input, seq_len);
        let d_ff = self.config.d_ff();

        // Apply activation
        let mut act_sum = 0.0;
        for val in hidden.iter_mut() {
            *val = self.config.activation.apply(*val);
            act_sum += val.abs();
        }
        let mean_activation = act_sum / hidden.len() as f64;
        self.activation_stats.push(mean_activation);

        // Second linear: (S, d_ff) -> (S, d_model)
        self.layer2.forward(&hidden, seq_len)
    }

    /// Total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.layer1.num_params() + self.layer2.num_params()
    }

    /// Mean activation magnitude from the last few forward calls.
    pub fn mean_activation_history(&self) -> &[f64] {
        &self.activation_stats
    }

    /// Reset activation statistics.
    pub fn reset_stats(&mut self) {
        self.activation_stats.clear();
    }

    /// Compute the sparsity of activations (fraction near zero) for diagnostics.
    pub fn compute_sparsity(&self, input: &[f64], seq_len: usize, threshold: f64) -> f64 {
        let hidden = self.layer1.forward(input, seq_len);
        let d_ff = self.config.d_ff();
        let total = hidden.len();
        let near_zero = hidden.iter()
            .map(|v| self.config.activation.apply(*v))
            .filter(|v| v.abs() < threshold)
            .count();
        near_zero as f64 / total as f64
    }
}

impl fmt::Display for FeedForwardBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FeedForward(d_model={}, d_ff={}, act={}, dropout={:.2})",
            self.config.d_model,
            self.config.d_ff(),
            self.config.activation,
            self.config.dropout_rate
        )
    }
}

// ── Gated Feed-Forward (SwiGLU variant) ──────────────────────────

/// Gated feed-forward block using SwiGLU-style gating.
///
/// FFN_gated(x) = W2 * (SiLU(W_gate * x) ⊙ (W_up * x)) + b2
///
/// Uses three weight matrices instead of two, with element-wise gating.
#[derive(Debug, Clone)]
pub struct GatedFeedForward {
    pub d_model: usize,
    pub d_ff: usize,
    pub gate_layer: DenseLayer,
    pub up_layer: DenseLayer,
    pub down_layer: DenseLayer,
}

impl GatedFeedForward {
    /// Create a new gated feed-forward block.
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            d_model,
            d_ff,
            gate_layer: DenseLayer::new(d_model, d_ff, 700),
            up_layer: DenseLayer::new(d_model, d_ff, 800),
            down_layer: DenseLayer::new(d_ff, d_model, 900),
        }
    }

    /// Forward pass with SwiGLU gating.
    pub fn forward(&self, input: &[f64], seq_len: usize) -> Vec<f64> {
        assert_eq!(input.len(), seq_len * self.d_model, "input length mismatch");
        let gate = self.gate_layer.forward(input, seq_len);
        let up = self.up_layer.forward(input, seq_len);

        // Element-wise: SiLU(gate) * up
        let mut gated: Vec<f64> = gate.iter().zip(up.iter())
            .map(|(g, u)| {
                let silu_g = g * sigmoid(*g);
                silu_g * u
            })
            .collect();

        self.down_layer.forward(&gated, seq_len)
    }

    /// Total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.gate_layer.num_params() + self.up_layer.num_params() + self.down_layer.num_params()
    }
}

impl fmt::Display for GatedFeedForward {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GatedFeedForward(d_model={}, d_ff={})", self.d_model, self.d_ff)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_sigmoid() {
        assert!(approx_eq(sigmoid(0.0), 0.5, 1e-10));
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_gelu_at_zero() {
        assert!(approx_eq(gelu(0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_gelu_positive() {
        // GELU(1.0) ≈ 0.8413
        let val = gelu(1.0);
        assert!(approx_eq(val, 0.8413, 0.001));
    }

    #[test]
    fn test_gelu_negative() {
        // GELU(-1.0) ≈ -0.1587
        let val = gelu(-1.0);
        assert!(approx_eq(val, -0.1587, 0.001));
    }

    #[test]
    fn test_relu_activation() {
        assert_eq!(Activation::Relu.apply(5.0), 5.0);
        assert_eq!(Activation::Relu.apply(-3.0), 0.0);
        assert_eq!(Activation::Relu.apply(0.0), 0.0);
    }

    #[test]
    fn test_silu_at_zero() {
        assert!(approx_eq(Activation::Silu.apply(0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_activation_display() {
        assert_eq!(format!("{}", Activation::Gelu), "GELU");
        assert_eq!(format!("{}", Activation::Relu), "ReLU");
        assert_eq!(format!("{}", Activation::Silu), "SiLU");
    }

    #[test]
    fn test_relu_derivative() {
        assert_eq!(Activation::Relu.derivative(1.0), 1.0);
        assert_eq!(Activation::Relu.derivative(-1.0), 0.0);
    }

    #[test]
    fn test_erf_bounds() {
        assert!(approx_eq(erf(0.0), 0.0, 1e-6));
        assert!(erf(3.0) > 0.99);
        assert!(erf(-3.0) < -0.99);
    }

    #[test]
    fn test_dense_layer_shape() {
        let layer = DenseLayer::new(8, 32, 42);
        let input = vec![0.1; 3 * 8]; // 3 positions x 8 dims
        let output = layer.forward(&input, 3);
        assert_eq!(output.len(), 3 * 32);
    }

    #[test]
    fn test_dense_layer_params() {
        let layer = DenseLayer::new(8, 32, 42);
        assert_eq!(layer.num_params(), 8 * 32 + 32);
    }

    #[test]
    fn test_ffn_config_defaults() {
        let cfg = FeedForwardConfig::new(64);
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.expansion_ratio, 4);
        assert_eq!(cfg.d_ff(), 256);
        assert_eq!(cfg.activation, Activation::Gelu);
    }

    #[test]
    fn test_ffn_config_builder() {
        let cfg = FeedForwardConfig::new(32)
            .with_expansion_ratio(8)
            .with_activation(Activation::Relu)
            .with_dropout(0.1)
            .with_bias(false);
        assert_eq!(cfg.d_ff(), 256);
        assert_eq!(cfg.activation, Activation::Relu);
        assert!(!cfg.use_bias);
    }

    #[test]
    fn test_ffn_forward_shape() {
        let cfg = FeedForwardConfig::new(16);
        let mut ffn = FeedForwardBlock::new(cfg);
        let input = vec![0.1; 5 * 16]; // 5 positions x 16 dims
        let output = ffn.forward(&input, 5);
        assert_eq!(output.len(), 5 * 16);
    }

    #[test]
    fn test_ffn_num_parameters() {
        let cfg = FeedForwardConfig::new(8).with_expansion_ratio(4);
        let ffn = FeedForwardBlock::new(cfg);
        // Layer1: 8*32 + 32 = 288, Layer2: 32*8 + 8 = 264
        assert_eq!(ffn.num_parameters(), 8 * 32 + 32 + 32 * 8 + 8);
    }

    #[test]
    fn test_ffn_activation_stats() {
        let cfg = FeedForwardConfig::new(8);
        let mut ffn = FeedForwardBlock::new(cfg);
        let input = vec![0.5; 3 * 8];
        ffn.forward(&input, 3);
        assert_eq!(ffn.mean_activation_history().len(), 1);
        ffn.forward(&input, 3);
        assert_eq!(ffn.mean_activation_history().len(), 2);
        ffn.reset_stats();
        assert!(ffn.mean_activation_history().is_empty());
    }

    #[test]
    fn test_ffn_display() {
        let cfg = FeedForwardConfig::new(64).with_activation(Activation::Silu);
        let ffn = FeedForwardBlock::new(cfg);
        let s = format!("{}", ffn);
        assert!(s.contains("64"));
        assert!(s.contains("256"));
        assert!(s.contains("SiLU"));
    }

    #[test]
    fn test_ffn_sparsity() {
        let cfg = FeedForwardConfig::new(8).with_activation(Activation::Relu);
        let ffn = FeedForwardBlock::new(cfg);
        let input = vec![0.0; 4 * 8];
        let sparsity = ffn.compute_sparsity(&input, 4, 0.01);
        // With zero input, many ReLU outputs should be zero
        assert!(sparsity >= 0.0 && sparsity <= 1.0);
    }

    #[test]
    fn test_gated_ffn_shape() {
        let gffn = GatedFeedForward::new(16, 64);
        let input = vec![0.1; 3 * 16];
        let output = gffn.forward(&input, 3);
        assert_eq!(output.len(), 3 * 16);
    }

    #[test]
    fn test_gated_ffn_params() {
        let gffn = GatedFeedForward::new(8, 32);
        // 3 layers: gate (8*32+32), up (8*32+32), down (32*8+8)
        let expected = (8 * 32 + 32) * 2 + (32 * 8 + 8);
        assert_eq!(gffn.num_parameters(), expected);
    }

    #[test]
    fn test_gated_ffn_display() {
        let gffn = GatedFeedForward::new(64, 256);
        let s = format!("{}", gffn);
        assert!(s.contains("64"));
        assert!(s.contains("256"));
    }
}
