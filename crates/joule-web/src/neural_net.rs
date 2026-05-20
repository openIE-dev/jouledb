//! Simple Feedforward Neural Network — dense layers, activations (ReLU,
//! sigmoid, tanh, softmax), backpropagation, SGD/Adam optimizers,
//! MSE/cross-entropy loss, batch training, model save/load as JSON.
//!
//! Pure Rust — no external ML or linear algebra dependencies.

use std::fmt;

use serde::{Deserialize, Serialize};

// ── Activation Functions ────────────────────────────────────────

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
}

impl Activation {
    /// Apply activation function to a vector.
    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        match self {
            Self::ReLU => z.iter().map(|x| x.max(0.0)).collect(),
            Self::Sigmoid => z.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
            Self::Tanh => z.iter().map(|x| x.tanh()).collect(),
            Self::Softmax => {
                let max_val = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = z.iter().map(|x| (x - max_val).exp()).collect();
                let sum: f64 = exps.iter().sum();
                exps.iter().map(|e| e / sum).collect()
            }
            Self::Linear => z.to_vec(),
        }
    }

    /// Compute activation derivative given the output of forward pass.
    /// For softmax, returns a placeholder (Jacobian handled in loss backprop).
    pub fn backward(&self, output: &[f64]) -> Vec<f64> {
        match self {
            Self::ReLU => output.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect(),
            Self::Sigmoid => output.iter().map(|x| x * (1.0 - x)).collect(),
            Self::Tanh => output.iter().map(|x| 1.0 - x * x).collect(),
            Self::Softmax => vec![1.0; output.len()], // handled in loss gradient
            Self::Linear => vec![1.0; output.len()],
        }
    }
}

// ── Loss Functions ──────────────────────────────────────────────

/// Loss function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFunction {
    MSE,
    CrossEntropy,
}

impl LossFunction {
    /// Compute loss value.
    pub fn compute(&self, predicted: &[f64], target: &[f64]) -> f64 {
        match self {
            Self::MSE => {
                let n = predicted.len() as f64;
                predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / n
            }
            Self::CrossEntropy => {
                let eps = 1e-15;
                -target
                    .iter()
                    .zip(predicted.iter())
                    .map(|(t, p)| t * (p.max(eps)).ln())
                    .sum::<f64>()
            }
        }
    }

    /// Compute gradient of loss with respect to predictions.
    pub fn gradient(&self, predicted: &[f64], target: &[f64]) -> Vec<f64> {
        match self {
            Self::MSE => {
                let n = predicted.len() as f64;
                predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| 2.0 * (p - t) / n)
                    .collect()
            }
            Self::CrossEntropy => {
                // For softmax + cross-entropy, the combined gradient is (predicted - target)
                predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| p - t)
                    .collect()
            }
        }
    }
}

// ── Dense Layer ─────────────────────────────────────────────────

/// A single dense (fully connected) layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    /// Weight matrix: weights[output][input].
    pub weights: Vec<Vec<f64>>,
    /// Bias vector.
    pub biases: Vec<f64>,
    /// Activation function.
    pub activation: Activation,
    /// Number of input features.
    pub n_input: usize,
    /// Number of output features.
    pub n_output: usize,
}

impl DenseLayer {
    /// Create a new dense layer with deterministic initialization.
    pub fn new(n_input: usize, n_output: usize, activation: Activation, seed: u64) -> Self {
        let mut rng_state = if seed == 0 { 1u64 } else { seed };
        let scale = (2.0 / n_input as f64).sqrt(); // He initialization scale

        let mut weights = Vec::with_capacity(n_output);
        for _ in 0..n_output {
            let mut row = Vec::with_capacity(n_input);
            for _ in 0..n_input {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let val = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                row.push(val * scale);
            }
            weights.push(row);
        }

        Self {
            weights,
            biases: vec![0.0; n_output],
            activation,
            n_input,
            n_output,
        }
    }

    /// Forward pass: compute output = activation(W * input + b).
    pub fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(input.len(), self.n_input, "input dimension mismatch");
        let mut z = vec![0.0; self.n_output];
        for i in 0..self.n_output {
            let mut sum = self.biases[i];
            for j in 0..self.n_input {
                sum += self.weights[i][j] * input[j];
            }
            z[i] = sum;
        }
        let output = self.activation.forward(&z);
        (z, output)
    }
}

// ── Optimizer ───────────────────────────────────────────────────

/// Optimizer type for training.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD { lr: f64, momentum: f64 },
    Adam { lr: f64, beta1: f64, beta2: f64, epsilon: f64 },
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::Adam { lr: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }
    }
}

/// Per-layer optimizer state for Adam.
#[derive(Debug, Clone)]
struct LayerOptimizerState {
    // SGD momentum
    weight_velocity: Vec<Vec<f64>>,
    bias_velocity: Vec<f64>,
    // Adam moments
    weight_m: Vec<Vec<f64>>,
    weight_v: Vec<Vec<f64>>,
    bias_m: Vec<f64>,
    bias_v: Vec<f64>,
}

impl LayerOptimizerState {
    fn new(n_output: usize, n_input: usize) -> Self {
        Self {
            weight_velocity: vec![vec![0.0; n_input]; n_output],
            bias_velocity: vec![0.0; n_output],
            weight_m: vec![vec![0.0; n_input]; n_output],
            weight_v: vec![vec![0.0; n_input]; n_output],
            bias_m: vec![0.0; n_output],
            bias_v: vec![0.0; n_output],
        }
    }
}

// ── Neural Network ──────────────────────────────────────────────

/// Simple feedforward neural network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNet {
    /// Network layers.
    pub layers: Vec<DenseLayer>,
    /// Loss function.
    pub loss_fn: LossFunction,
    /// Optimizer configuration.
    pub optimizer: OptimizerType,
}

/// Training history for a single epoch.
#[derive(Debug, Clone)]
pub struct TrainHistory {
    /// Loss values per epoch.
    pub losses: Vec<f64>,
}

impl NeuralNet {
    /// Create a new neural network from layer specifications.
    ///
    /// `layer_sizes` defines the number of neurons in each layer including input.
    /// `activations` has one entry per hidden/output layer.
    pub fn new(
        layer_sizes: &[usize],
        activations: &[Activation],
        loss_fn: LossFunction,
        optimizer: OptimizerType,
        seed: u64,
    ) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least input and output layer");
        assert_eq!(
            activations.len(),
            layer_sizes.len() - 1,
            "need one activation per layer transition"
        );

        let mut layers = Vec::with_capacity(activations.len());
        for i in 0..activations.len() {
            let layer_seed = seed.wrapping_add(i as u64 * 1000);
            layers.push(DenseLayer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                activations[i],
                layer_seed,
            ));
        }

        Self { layers, loss_fn, optimizer }
    }

    /// Forward pass through the network.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            let (_, output) = layer.forward(&current);
            current = output;
        }
        current
    }

    /// Predict output for a batch of inputs.
    pub fn predict_batch(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        inputs.iter().map(|x| self.forward(x)).collect()
    }

    /// Train the network on a single sample (online learning).
    /// Returns the loss for this sample.
    pub fn train_sample(
        &mut self,
        input: &[f64],
        target: &[f64],
        opt_states: &mut [LayerOptimizerState],
        t: usize,
    ) -> f64 {
        // Forward pass — store all intermediate values
        let n_layers = self.layers.len();
        let mut layer_inputs: Vec<Vec<f64>> = Vec::with_capacity(n_layers + 1);
        let mut layer_z_vals: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
        let mut layer_outputs: Vec<Vec<f64>> = Vec::with_capacity(n_layers);

        layer_inputs.push(input.to_vec());
        let mut current = input.to_vec();
        for layer in &self.layers {
            let (z, output) = layer.forward(&current);
            layer_z_vals.push(z);
            layer_outputs.push(output.clone());
            current = output;
            layer_inputs.push(current.clone());
        }

        let predicted = &layer_outputs[n_layers - 1];
        let loss = self.loss_fn.compute(predicted, target);

        // Backward pass
        let mut delta = self.loss_fn.gradient(predicted, target);

        // For MSE with non-softmax output, multiply by activation derivative
        let last_activation = self.layers[n_layers - 1].activation;
        if self.loss_fn == LossFunction::MSE || last_activation != Activation::Softmax {
            if last_activation != Activation::Softmax {
                let act_deriv = last_activation.backward(&layer_outputs[n_layers - 1]);
                for (d, ad) in delta.iter_mut().zip(act_deriv.iter()) {
                    *d *= ad;
                }
            }
        }

        for l in (0..n_layers).rev() {
            let layer_input = &layer_inputs[l];
            let n_out = self.layers[l].n_output;
            let n_in = self.layers[l].n_input;

            // Compute weight and bias gradients
            let mut weight_grad = vec![vec![0.0; n_in]; n_out];
            for i in 0..n_out {
                for j in 0..n_in {
                    weight_grad[i][j] = delta[i] * layer_input[j];
                }
            }
            let bias_grad: Vec<f64> = delta.clone();

            // Compute delta for previous layer
            if l > 0 {
                let mut new_delta = vec![0.0; n_in];
                for j in 0..n_in {
                    let mut sum = 0.0;
                    for i in 0..n_out {
                        sum += self.layers[l].weights[i][j] * delta[i];
                    }
                    new_delta[j] = sum;
                }
                let act_deriv = self.layers[l - 1].activation.backward(&layer_outputs[l - 1]);
                for (d, ad) in new_delta.iter_mut().zip(act_deriv.iter()) {
                    *d *= ad;
                }
                delta = new_delta;
            }

            // Apply optimizer
            self.apply_optimizer(l, &weight_grad, &bias_grad, opt_states, t);
        }

        loss
    }

    fn apply_optimizer(
        &mut self,
        layer_idx: usize,
        weight_grad: &[Vec<f64>],
        bias_grad: &[f64],
        opt_states: &mut [LayerOptimizerState],
        t: usize,
    ) {
        let state = &mut opt_states[layer_idx];
        match self.optimizer {
            OptimizerType::SGD { lr, momentum } => {
                for i in 0..self.layers[layer_idx].n_output {
                    for j in 0..self.layers[layer_idx].n_input {
                        state.weight_velocity[i][j] =
                            momentum * state.weight_velocity[i][j] + lr * weight_grad[i][j];
                        self.layers[layer_idx].weights[i][j] -= state.weight_velocity[i][j];
                    }
                    state.bias_velocity[i] = momentum * state.bias_velocity[i] + lr * bias_grad[i];
                    self.layers[layer_idx].biases[i] -= state.bias_velocity[i];
                }
            }
            OptimizerType::Adam { lr, beta1, beta2, epsilon } => {
                let t_f = (t + 1) as f64;
                for i in 0..self.layers[layer_idx].n_output {
                    for j in 0..self.layers[layer_idx].n_input {
                        state.weight_m[i][j] =
                            beta1 * state.weight_m[i][j] + (1.0 - beta1) * weight_grad[i][j];
                        state.weight_v[i][j] = beta2 * state.weight_v[i][j]
                            + (1.0 - beta2) * weight_grad[i][j] * weight_grad[i][j];
                        let m_hat = state.weight_m[i][j] / (1.0 - beta1.powf(t_f));
                        let v_hat = state.weight_v[i][j] / (1.0 - beta2.powf(t_f));
                        self.layers[layer_idx].weights[i][j] -=
                            lr * m_hat / (v_hat.sqrt() + epsilon);
                    }
                    state.bias_m[i] = beta1 * state.bias_m[i] + (1.0 - beta1) * bias_grad[i];
                    state.bias_v[i] =
                        beta2 * state.bias_v[i] + (1.0 - beta2) * bias_grad[i] * bias_grad[i];
                    let m_hat = state.bias_m[i] / (1.0 - beta1.powf(t_f));
                    let v_hat = state.bias_v[i] / (1.0 - beta2.powf(t_f));
                    self.layers[layer_idx].biases[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        }
    }

    /// Train on a dataset for a number of epochs.
    /// Returns training history with loss per epoch.
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        epochs: usize,
    ) -> TrainHistory {
        assert_eq!(inputs.len(), targets.len(), "input/target count mismatch");

        let mut opt_states: Vec<LayerOptimizerState> = self
            .layers
            .iter()
            .map(|l| LayerOptimizerState::new(l.n_output, l.n_input))
            .collect();

        let mut losses = Vec::with_capacity(epochs);
        let mut global_step = 0;

        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                epoch_loss += self.train_sample(input, target, &mut opt_states, global_step);
                global_step += 1;
            }
            losses.push(epoch_loss / inputs.len() as f64);
        }

        TrainHistory { losses }
    }

    /// Compute average loss on a dataset.
    pub fn evaluate(&self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
        assert_eq!(inputs.len(), targets.len());
        if inputs.is_empty() {
            return 0.0;
        }
        let total: f64 = inputs
            .iter()
            .zip(targets.iter())
            .map(|(x, t)| {
                let pred = self.forward(x);
                self.loss_fn.compute(&pred, t)
            })
            .sum();
        total / inputs.len() as f64
    }

    /// Return the total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.n_input * l.n_output + l.n_output)
            .sum()
    }

    /// Return layer sizes.
    pub fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![self.layers[0].n_input];
        for layer in &self.layers {
            sizes.push(layer.n_output);
        }
        sizes
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for NeuralNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sizes = self.layer_sizes();
        let acts: Vec<String> = self.layers.iter().map(|l| format!("{:?}", l.activation)).collect();
        write!(
            f,
            "NeuralNet(layers={:?}, activations=[{}], params={})",
            sizes,
            acts.join(", "),
            self.n_params()
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward() {
        let out = Activation::ReLU.forward(&[-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let out = Activation::Sigmoid.forward(&[0.0]);
        assert!((out[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_forward() {
        let out = Activation::Tanh.forward(&[0.0]);
        assert!(out[0].abs() < 1e-10);
    }

    #[test]
    fn test_softmax_forward() {
        let out = Activation::Softmax.forward(&[1.0, 2.0, 3.0]);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(out[2] > out[1]);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn test_relu_backward() {
        let deriv = Activation::ReLU.backward(&[0.0, 0.5, -0.1, 1.0]);
        assert_eq!(deriv, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_backward() {
        let deriv = Activation::Sigmoid.backward(&[0.5]);
        assert!((deriv[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss() {
        let loss = LossFunction::MSE.compute(&[1.0, 2.0], &[1.0, 2.0]);
        assert!(loss.abs() < 1e-10);

        let loss2 = LossFunction::MSE.compute(&[1.0, 2.0], &[2.0, 3.0]);
        assert!((loss2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let loss = LossFunction::CrossEntropy.compute(&[0.9, 0.1], &[1.0, 0.0]);
        assert!(loss > 0.0);
        assert!(loss < 1.0);
    }

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer::new(2, 3, Activation::ReLU, 42);
        let (_, output) = layer.forward(&[1.0, 2.0]);
        assert_eq!(output.len(), 3);
        // ReLU output should be non-negative
        for v in &output {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_network_forward() {
        let net = NeuralNet::new(
            &[2, 4, 2],
            &[Activation::ReLU, Activation::Softmax],
            LossFunction::CrossEntropy,
            OptimizerType::default(),
            42,
        );
        let output = net.forward(&[1.0, 2.0]);
        assert_eq!(output.len(), 2);
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_network_train_xor() {
        let mut net = NeuralNet::new(
            &[2, 8, 1],
            &[Activation::Tanh, Activation::Sigmoid],
            LossFunction::MSE,
            OptimizerType::SGD { lr: 0.5, momentum: 0.0 },
            42,
        );

        let inputs = vec![
            vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0], vec![1.0], vec![1.0], vec![0.0],
        ];

        let history = net.train(&inputs, &targets, 500);
        assert!(!history.losses.is_empty());
        // Loss should decrease
        assert!(history.losses.last().unwrap() < history.losses.first().unwrap());
    }

    #[test]
    fn test_network_evaluate() {
        let net = NeuralNet::new(
            &[2, 4, 2],
            &[Activation::ReLU, Activation::Softmax],
            LossFunction::CrossEntropy,
            OptimizerType::default(),
            42,
        );
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let loss = net.evaluate(&inputs, &targets);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_n_params() {
        let net = NeuralNet::new(
            &[2, 4, 3],
            &[Activation::ReLU, Activation::Softmax],
            LossFunction::CrossEntropy,
            OptimizerType::default(),
            42,
        );
        // Layer 0: 2*4 + 4 = 12, Layer 1: 4*3 + 3 = 15, total = 27
        assert_eq!(net.n_params(), 27);
    }

    #[test]
    fn test_layer_sizes() {
        let net = NeuralNet::new(
            &[3, 5, 2],
            &[Activation::ReLU, Activation::Sigmoid],
            LossFunction::MSE,
            OptimizerType::default(),
            42,
        );
        assert_eq!(net.layer_sizes(), vec![3, 5, 2]);
    }

    #[test]
    fn test_predict_batch() {
        let net = NeuralNet::new(
            &[2, 3, 1],
            &[Activation::ReLU, Activation::Sigmoid],
            LossFunction::MSE,
            OptimizerType::default(),
            42,
        );
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let outputs = net.predict_batch(&inputs);
        assert_eq!(outputs.len(), 3);
        for out in &outputs {
            assert_eq!(out.len(), 1);
        }
    }

    #[test]
    fn test_serialization() {
        let net = NeuralNet::new(
            &[2, 3, 1],
            &[Activation::ReLU, Activation::Sigmoid],
            LossFunction::MSE,
            OptimizerType::default(),
            42,
        );
        let json = net.to_json().unwrap();
        let net2 = NeuralNet::from_json(&json).unwrap();
        let out1 = net.forward(&[1.0, 2.0]);
        let out2 = net2.forward(&[1.0, 2.0]);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_display() {
        let net = NeuralNet::new(
            &[2, 4, 2],
            &[Activation::ReLU, Activation::Softmax],
            LossFunction::CrossEntropy,
            OptimizerType::default(),
            42,
        );
        let s = format!("{}", net);
        assert!(s.contains("NeuralNet"));
        assert!(s.contains("ReLU"));
    }

    #[test]
    fn test_sgd_optimizer() {
        let mut net = NeuralNet::new(
            &[1, 4, 1],
            &[Activation::Tanh, Activation::Linear],
            LossFunction::MSE,
            OptimizerType::SGD { lr: 0.01, momentum: 0.9 },
            42,
        );
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![vec![2.0], vec![4.0], vec![6.0]];
        let history = net.train(&inputs, &targets, 100);
        assert!(history.losses.last().unwrap() < history.losses.first().unwrap());
    }

    #[test]
    fn test_linear_activation() {
        let out = Activation::Linear.forward(&[-1.0, 0.5, 2.0]);
        assert_eq!(out, vec![-1.0, 0.5, 2.0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let out = Activation::Softmax.forward(&[1000.0, 1001.0, 1002.0]);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
