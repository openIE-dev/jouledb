//! Fully-connected (dense) neural network layer.
//!
//! Implements a linear transformation `y = activation(Wx + b)` with
//! configurable weight initialization, bias terms, and activation
//! functions. Supports forward pass, gradient computation, and
//! parameter updates via simple SGD.

use std::fmt;

// ── Activation ────────────────────────────────────────────────────

/// Activation function applied after the linear transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DenseActivation {
    None,
    Relu,
    Sigmoid,
    Tanh,
}

impl DenseActivation {
    /// Apply the activation element-wise.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::None => x,
            Self::Relu => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
        }
    }

    /// Derivative of the activation given the *output* value.
    pub fn derivative(&self, output: f64) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Relu => if output > 0.0 { 1.0 } else { 0.0 },
            Self::Sigmoid => output * (1.0 - output),
            Self::Tanh => 1.0 - output * output,
        }
    }
}

impl fmt::Display for DenseActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Relu => write!(f, "relu"),
            Self::Sigmoid => write!(f, "sigmoid"),
            Self::Tanh => write!(f, "tanh"),
        }
    }
}

// ── Weight Initialization ─────────────────────────────────────────

/// Strategy for initializing weight matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightInit {
    /// All zeros.
    Zeros,
    /// Uniform in [-bound, bound] where bound = 1/sqrt(fan_in).
    XavierUniform,
    /// Small constant value.
    Constant(f64),
    /// He initialization scaled by sqrt(2/fan_in).
    HeUniform,
}

// ── Simple LCG PRNG ──────────────────────────────────────────────

/// Minimal linear congruential generator for weight init.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [-bound, bound).
    fn uniform(&mut self, bound: f64) -> f64 {
        self.next_f64() * 2.0 * bound - bound
    }
}

// ── DenseLayer ────────────────────────────────────────────────────

/// A fully-connected layer: `y = activation(W * x + b)`.
///
/// Weights are stored in row-major order: `weights[out_i * in_features + in_j]`.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub activation: DenseActivation,
    pub use_bias: bool,
    last_input: Vec<f64>,
    last_pre_activation: Vec<f64>,
    last_output: Vec<f64>,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier uniform initialization.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = Lcg::new(in_features as u64 ^ (out_features as u64).wrapping_mul(31));
        let bound = 1.0 / (in_features as f64).sqrt();
        let weights: Vec<f64> = (0..in_features * out_features)
            .map(|_| rng.uniform(bound))
            .collect();
        let biases = vec![0.0; out_features];

        Self {
            in_features,
            out_features,
            weights,
            biases,
            activation: DenseActivation::None,
            use_bias: true,
            last_input: Vec::new(),
            last_pre_activation: Vec::new(),
            last_output: Vec::new(),
        }
    }

    /// Set the activation function.
    pub fn with_activation(mut self, act: DenseActivation) -> Self {
        self.activation = act;
        self
    }

    /// Disable bias terms.
    pub fn with_no_bias(mut self) -> Self {
        self.use_bias = false;
        self.biases = vec![0.0; self.out_features];
        self
    }

    /// Re-initialize weights with a specific strategy.
    pub fn with_init(mut self, init: WeightInit, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let fan_in = self.in_features as f64;
        match init {
            WeightInit::Zeros => {
                self.weights.iter_mut().for_each(|w| *w = 0.0);
            }
            WeightInit::XavierUniform => {
                let bound = 1.0 / fan_in.sqrt();
                self.weights.iter_mut().for_each(|w| *w = rng.uniform(bound));
            }
            WeightInit::Constant(c) => {
                self.weights.iter_mut().for_each(|w| *w = c);
            }
            WeightInit::HeUniform => {
                let bound = (2.0 / fan_in).sqrt();
                self.weights.iter_mut().for_each(|w| *w = rng.uniform(bound));
            }
        }
        self
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        let w = self.in_features * self.out_features;
        if self.use_bias { w + self.out_features } else { w }
    }

    /// Forward pass for a single sample.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.in_features, "input size mismatch");
        self.last_input = input.to_vec();

        let mut pre_act = vec![0.0; self.out_features];
        for o in 0..self.out_features {
            let mut sum = if self.use_bias { self.biases[o] } else { 0.0 };
            let row_start = o * self.in_features;
            for i in 0..self.in_features {
                sum += self.weights[row_start + i] * input[i];
            }
            pre_act[o] = sum;
        }

        self.last_pre_activation = pre_act.clone();
        let output: Vec<f64> = pre_act.iter().map(|v| self.activation.apply(*v)).collect();
        self.last_output = output.clone();
        output
    }

    /// Batch forward: process multiple samples, returns flattened outputs.
    pub fn forward_batch(&mut self, inputs: &[f64], batch_size: usize) -> Vec<f64> {
        assert_eq!(inputs.len(), batch_size * self.in_features);
        let mut all_outputs = Vec::with_capacity(batch_size * self.out_features);
        for b in 0..batch_size {
            let start = b * self.in_features;
            let end = start + self.in_features;
            let out = self.forward(&inputs[start..end]);
            all_outputs.extend_from_slice(&out);
        }
        all_outputs
    }

    /// Backward pass: given upstream gradient, returns gradient w.r.t. input.
    /// Also accumulates weight and bias gradients into the provided accumulators.
    pub fn backward(
        &self,
        grad_output: &[f64],
        grad_weights: &mut [f64],
        grad_biases: &mut [f64],
    ) -> Vec<f64> {
        assert_eq!(grad_output.len(), self.out_features);
        assert_eq!(grad_weights.len(), self.weights.len());
        assert_eq!(grad_biases.len(), self.biases.len());

        // Apply activation derivative
        let mut delta = vec![0.0; self.out_features];
        for o in 0..self.out_features {
            delta[o] = grad_output[o] * self.activation.derivative(self.last_output[o]);
        }

        // Accumulate weight gradients: dW[o][i] += delta[o] * input[i]
        for o in 0..self.out_features {
            let row_start = o * self.in_features;
            for i in 0..self.in_features {
                grad_weights[row_start + i] += delta[o] * self.last_input[i];
            }
            grad_biases[o] += delta[o];
        }

        // Compute input gradient: dx[i] = sum_o W[o][i] * delta[o]
        let mut grad_input = vec![0.0; self.in_features];
        for o in 0..self.out_features {
            let row_start = o * self.in_features;
            for i in 0..self.in_features {
                grad_input[i] += self.weights[row_start + i] * delta[o];
            }
        }

        grad_input
    }

    /// Apply simple SGD update to weights and biases.
    pub fn sgd_update(&mut self, grad_weights: &[f64], grad_biases: &[f64], lr: f64) {
        for (w, gw) in self.weights.iter_mut().zip(grad_weights.iter()) {
            *w -= lr * gw;
        }
        if self.use_bias {
            for (b, gb) in self.biases.iter_mut().zip(grad_biases.iter()) {
                *b -= lr * gb;
            }
        }
    }

    /// L2 norm of all weight parameters (for regularization).
    pub fn weight_l2_norm(&self) -> f64 {
        self.weights.iter().map(|w| w * w).sum::<f64>().sqrt()
    }

    /// Flatten all parameters into a single vector.
    pub fn flatten_params(&self) -> Vec<f64> {
        let mut params = self.weights.clone();
        if self.use_bias {
            params.extend_from_slice(&self.biases);
        }
        params
    }

    /// Load parameters from a flat vector.
    pub fn load_params(&mut self, params: &[f64]) {
        let w_count = self.in_features * self.out_features;
        assert!(params.len() >= w_count);
        self.weights.copy_from_slice(&params[..w_count]);
        if self.use_bias && params.len() >= w_count + self.out_features {
            self.biases.copy_from_slice(&params[w_count..w_count + self.out_features]);
        }
    }
}

impl fmt::Display for DenseLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenseLayer({} -> {}, act={}, bias={}, params={})",
            self.in_features,
            self.out_features,
            self.activation,
            self.use_bias,
            self.param_count()
        )
    }
}

// ── Two-Layer MLP ─────────────────────────────────────────────────

/// A simple two-layer MLP built from DenseLayers for demonstration.
#[derive(Debug, Clone)]
pub struct TwoLayerMlp {
    pub hidden: DenseLayer,
    pub output: DenseLayer,
}

impl TwoLayerMlp {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            hidden: DenseLayer::new(input_dim, hidden_dim)
                .with_activation(DenseActivation::Relu),
            output: DenseLayer::new(hidden_dim, output_dim),
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let h = self.hidden.forward(input);
        self.output.forward(&h)
    }

    pub fn param_count(&self) -> usize {
        self.hidden.param_count() + self.output.param_count()
    }
}

impl fmt::Display for TwoLayerMlp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TwoLayerMlp({}, {})", self.hidden, self.output)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer_creation() {
        let layer = DenseLayer::new(4, 3);
        assert_eq!(layer.in_features, 4);
        assert_eq!(layer.out_features, 3);
        assert_eq!(layer.weights.len(), 12);
        assert_eq!(layer.biases.len(), 3);
    }

    #[test]
    fn test_param_count_with_bias() {
        let layer = DenseLayer::new(10, 5);
        assert_eq!(layer.param_count(), 55); // 10*5 + 5
    }

    #[test]
    fn test_param_count_no_bias() {
        let layer = DenseLayer::new(10, 5).with_no_bias();
        assert_eq!(layer.param_count(), 50);
    }

    #[test]
    fn test_forward_identity() {
        let mut layer = DenseLayer::new(3, 2)
            .with_init(WeightInit::Zeros, 0)
            .with_no_bias();
        let out = layer.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn test_forward_constant_weights() {
        let mut layer = DenseLayer::new(3, 1)
            .with_init(WeightInit::Constant(1.0), 0)
            .with_no_bias();
        let out = layer.forward(&[1.0, 2.0, 3.0]);
        assert!((out[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu_activation() {
        let act = DenseActivation::Relu;
        assert_eq!(act.apply(5.0), 5.0);
        assert_eq!(act.apply(-3.0), 0.0);
        assert_eq!(act.derivative(5.0), 1.0);
        assert_eq!(act.derivative(-1.0), 0.0);
    }

    #[test]
    fn test_sigmoid_activation() {
        let act = DenseActivation::Sigmoid;
        let y = act.apply(0.0);
        assert!((y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_activation() {
        let act = DenseActivation::Tanh;
        let y = act.apply(0.0);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn test_forward_output_size() {
        let mut layer = DenseLayer::new(5, 3);
        let out = layer.forward(&[1.0, 0.0, -1.0, 2.0, 0.5]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_batch_forward() {
        let mut layer = DenseLayer::new(2, 3);
        let inputs = vec![1.0, 2.0, 3.0, 4.0];
        let out = layer.forward_batch(&inputs, 2);
        assert_eq!(out.len(), 6); // 2 samples * 3 outputs
    }

    #[test]
    fn test_backward_gradient_shape() {
        let mut layer = DenseLayer::new(3, 2);
        layer.forward(&[1.0, 2.0, 3.0]);
        let mut gw = vec![0.0; 6];
        let mut gb = vec![0.0; 2];
        let gi = layer.backward(&[1.0, -1.0], &mut gw, &mut gb);
        assert_eq!(gi.len(), 3);
    }

    #[test]
    fn test_sgd_update_changes_weights() {
        let mut layer = DenseLayer::new(2, 1)
            .with_init(WeightInit::Constant(1.0), 0);
        let gw = vec![0.1, 0.2];
        let gb = vec![0.05];
        layer.sgd_update(&gw, &gb, 1.0);
        assert!((layer.weights[0] - 0.9).abs() < 1e-10);
        assert!((layer.weights[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_weight_l2_norm() {
        let layer = DenseLayer::new(2, 1)
            .with_init(WeightInit::Constant(3.0), 0);
        let norm = layer.weight_l2_norm();
        // sqrt(3^2 + 3^2) = sqrt(18)
        assert!((norm - (18.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_flatten_and_load_params() {
        let layer = DenseLayer::new(3, 2);
        let params = layer.flatten_params();
        assert_eq!(params.len(), 8); // 6 weights + 2 biases
        let mut layer2 = DenseLayer::new(3, 2);
        layer2.load_params(&params);
        assert_eq!(layer2.weights, layer.weights);
        assert_eq!(layer2.biases, layer.biases);
    }

    #[test]
    fn test_he_init_bounds() {
        let layer = DenseLayer::new(100, 10).with_init(WeightInit::HeUniform, 42);
        let bound = (2.0 / 100.0_f64).sqrt();
        for &w in &layer.weights {
            assert!(w.abs() <= bound + 1e-10);
        }
    }

    #[test]
    fn test_xavier_init_bounds() {
        let layer = DenseLayer::new(64, 32).with_init(WeightInit::XavierUniform, 99);
        let bound = 1.0 / (64.0_f64).sqrt();
        for &w in &layer.weights {
            assert!(w.abs() <= bound + 1e-10);
        }
    }

    #[test]
    fn test_display() {
        let layer = DenseLayer::new(4, 3).with_activation(DenseActivation::Relu);
        let s = format!("{}", layer);
        assert!(s.contains("DenseLayer(4 -> 3"));
        assert!(s.contains("relu"));
    }

    #[test]
    fn test_two_layer_mlp() {
        let mut mlp = TwoLayerMlp::new(4, 8, 2);
        let out = mlp.forward(&[1.0, 0.0, -1.0, 0.5]);
        assert_eq!(out.len(), 2);
        assert_eq!(mlp.param_count(), (4 * 8 + 8) + (8 * 2 + 2));
    }

    #[test]
    fn test_activation_display() {
        assert_eq!(format!("{}", DenseActivation::Relu), "relu");
        assert_eq!(format!("{}", DenseActivation::None), "none");
        assert_eq!(format!("{}", DenseActivation::Sigmoid), "sigmoid");
        assert_eq!(format!("{}", DenseActivation::Tanh), "tanh");
    }

    #[test]
    fn test_backward_zero_gradient() {
        let mut layer = DenseLayer::new(3, 2)
            .with_init(WeightInit::Constant(1.0), 0);
        layer.forward(&[1.0, 2.0, 3.0]);
        let mut gw = vec![0.0; 6];
        let mut gb = vec![0.0; 2];
        let gi = layer.backward(&[0.0, 0.0], &mut gw, &mut gb);
        assert!(gi.iter().all(|v| v.abs() < 1e-10));
        assert!(gw.iter().all(|v| v.abs() < 1e-10));
    }
}
