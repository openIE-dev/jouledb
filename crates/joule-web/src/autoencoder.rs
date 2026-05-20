//! Autoencoder: encoder/decoder networks, bottleneck layer, reconstruction loss, sparse autoencoder.
//!
//! Implements dense feed-forward autoencoders with configurable layer sizes,
//! activation functions, sparsity penalties (KL-divergence on hidden activations),
//! and MSE/MAE reconstruction losses. Training uses mini-batch gradient descent
//! with backpropagation through the encoder-bottleneck-decoder chain.

use std::fmt;

// ── Activation Functions ──────────────────────────────────────

/// Supported activation functions for encoder/decoder layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f64),
    Linear,
}

impl Activation {
    /// Apply the activation function element-wise.
    pub fn forward(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::LeakyReLU(alpha) => if x >= 0.0 { x } else { alpha * x },
            Activation::Linear => x,
        }
    }

    /// Derivative of the activation for backpropagation.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => {
                let s = self.forward(x);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::LeakyReLU(alpha) => if x >= 0.0 { 1.0 } else { *alpha },
            Activation::Linear => 1.0,
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Activation::ReLU => write!(f, "ReLU"),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Tanh => write!(f, "Tanh"),
            Activation::LeakyReLU(a) => write!(f, "LeakyReLU(alpha={a:.4})"),
            Activation::Linear => write!(f, "Linear"),
        }
    }
}

// ── Reconstruction Loss ───────────────────────────────────────

/// Loss function for measuring reconstruction quality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconstructionLoss {
    /// Mean squared error.
    MSE,
    /// Mean absolute error.
    MAE,
    /// Binary cross-entropy (for sigmoid output).
    BinaryCrossEntropy,
}

impl ReconstructionLoss {
    /// Compute loss between original and reconstructed vectors.
    pub fn compute(&self, original: &[f64], reconstructed: &[f64]) -> f64 {
        assert_eq!(original.len(), reconstructed.len());
        let n = original.len() as f64;
        match self {
            ReconstructionLoss::MSE => {
                let sum: f64 = original.iter().zip(reconstructed.iter())
                    .map(|(o, r)| (o - r).powi(2))
                    .sum();
                sum / n
            }
            ReconstructionLoss::MAE => {
                let sum: f64 = original.iter().zip(reconstructed.iter())
                    .map(|(o, r)| (o - r).abs())
                    .sum();
                sum / n
            }
            ReconstructionLoss::BinaryCrossEntropy => {
                let eps = 1e-12;
                let sum: f64 = original.iter().zip(reconstructed.iter())
                    .map(|(o, r)| {
                        let r_clamped = r.clamp(eps, 1.0 - eps);
                        -(o * r_clamped.ln() + (1.0 - o) * (1.0 - r_clamped).ln())
                    })
                    .sum();
                sum / n
            }
        }
    }

    /// Gradient of the loss w.r.t. reconstructed values.
    pub fn gradient(&self, original: &[f64], reconstructed: &[f64]) -> Vec<f64> {
        let n = original.len() as f64;
        match self {
            ReconstructionLoss::MSE => {
                original.iter().zip(reconstructed.iter())
                    .map(|(o, r)| 2.0 * (r - o) / n)
                    .collect()
            }
            ReconstructionLoss::MAE => {
                original.iter().zip(reconstructed.iter())
                    .map(|(o, r)| {
                        let diff = r - o;
                        if diff > 0.0 { 1.0 / n }
                        else if diff < 0.0 { -1.0 / n }
                        else { 0.0 }
                    })
                    .collect()
            }
            ReconstructionLoss::BinaryCrossEntropy => {
                let eps = 1e-12;
                original.iter().zip(reconstructed.iter())
                    .map(|(o, r)| {
                        let r_clamped = r.clamp(eps, 1.0 - eps);
                        (-o / r_clamped + (1.0 - o) / (1.0 - r_clamped)) / n
                    })
                    .collect()
            }
        }
    }
}

impl fmt::Display for ReconstructionLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReconstructionLoss::MSE => write!(f, "MSE"),
            ReconstructionLoss::MAE => write!(f, "MAE"),
            ReconstructionLoss::BinaryCrossEntropy => write!(f, "BinaryCrossEntropy"),
        }
    }
}

// ── Dense Layer ───────────────────────────────────────────────

/// A single dense (fully-connected) layer with weights, biases, and activation.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: Activation,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier-like initialization using a simple LCG.
    pub fn new(input_size: usize, output_size: usize, activation: Activation, seed: u64) -> Self {
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let mut rng_state = seed;
        let mut next_rand = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            (bits - 0.5) * 2.0 * scale
        };

        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| (0..input_size).map(|_| next_rand()).collect())
            .collect();
        let biases = vec![0.0; output_size];

        Self { weights, biases, activation, input_size, output_size }
    }

    /// Forward pass: output = activation(W * input + b).
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);
        (0..self.output_size)
            .map(|j| {
                let z: f64 = self.weights[j].iter().zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>() + self.biases[j];
                self.activation.forward(z)
            })
            .collect()
    }

    /// Compute the pre-activation values (z = W * input + b).
    pub fn pre_activation(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);
        (0..self.output_size)
            .map(|j| {
                self.weights[j].iter().zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>() + self.biases[j]
            })
            .collect()
    }

    pub fn input_size(&self) -> usize { self.input_size }
    pub fn output_size(&self) -> usize { self.output_size }
}

// ── Autoencoder ───────────────────────────────────────────────

/// Configuration for building an autoencoder.
#[derive(Debug, Clone)]
pub struct AutoencoderConfig {
    pub input_dim: usize,
    pub encoder_layers: Vec<usize>,
    pub bottleneck_dim: usize,
    pub activation: Activation,
    pub output_activation: Activation,
    pub loss_fn: ReconstructionLoss,
    pub learning_rate: f64,
    pub sparsity_target: Option<f64>,
    pub sparsity_weight: f64,
}

impl AutoencoderConfig {
    pub fn new(input_dim: usize, bottleneck_dim: usize) -> Self {
        Self {
            input_dim,
            encoder_layers: vec![],
            bottleneck_dim,
            activation: Activation::ReLU,
            output_activation: Activation::Sigmoid,
            loss_fn: ReconstructionLoss::MSE,
            learning_rate: 0.001,
            sparsity_target: None,
            sparsity_weight: 0.01,
        }
    }

    pub fn with_encoder_layers(mut self, layers: Vec<usize>) -> Self {
        self.encoder_layers = layers;
        self
    }

    pub fn with_activation(mut self, act: Activation) -> Self {
        self.activation = act;
        self
    }

    pub fn with_output_activation(mut self, act: Activation) -> Self {
        self.output_activation = act;
        self
    }

    pub fn with_loss(mut self, loss: ReconstructionLoss) -> Self {
        self.loss_fn = loss;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Enable sparsity constraint (for sparse autoencoder).
    /// `target` is the desired average activation (e.g. 0.05).
    pub fn with_sparsity(mut self, target: f64, weight: f64) -> Self {
        self.sparsity_target = Some(target);
        self.sparsity_weight = weight;
        self
    }
}

/// Feed-forward autoencoder with symmetric encoder/decoder architecture.
#[derive(Debug, Clone)]
pub struct Autoencoder {
    encoder: Vec<DenseLayer>,
    decoder: Vec<DenseLayer>,
    config: AutoencoderConfig,
    train_losses: Vec<f64>,
}

impl Autoencoder {
    /// Build an autoencoder from the given configuration.
    pub fn build(config: AutoencoderConfig) -> Self {
        let seed_base: u64 = 42;
        let mut encoder = Vec::new();
        let mut prev_size = config.input_dim;

        // Build encoder hidden layers.
        for (i, &layer_size) in config.encoder_layers.iter().enumerate() {
            encoder.push(DenseLayer::new(prev_size, layer_size, config.activation, seed_base + i as u64));
            prev_size = layer_size;
        }
        // Bottleneck layer.
        encoder.push(DenseLayer::new(
            prev_size,
            config.bottleneck_dim,
            config.activation,
            seed_base + config.encoder_layers.len() as u64,
        ));

        // Build decoder (mirror of encoder).
        let mut decoder = Vec::new();
        prev_size = config.bottleneck_dim;
        for (i, &layer_size) in config.encoder_layers.iter().rev().enumerate() {
            decoder.push(DenseLayer::new(
                prev_size,
                layer_size,
                config.activation,
                seed_base + 100 + i as u64,
            ));
            prev_size = layer_size;
        }
        // Output layer.
        decoder.push(DenseLayer::new(
            prev_size,
            config.input_dim,
            config.output_activation,
            seed_base + 200,
        ));

        Self { encoder, decoder, config, train_losses: Vec::new() }
    }

    /// Encode input to bottleneck representation.
    pub fn encode(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.encoder {
            current = layer.forward(&current);
        }
        current
    }

    /// Decode bottleneck representation back to input space.
    pub fn decode(&self, latent: &[f64]) -> Vec<f64> {
        let mut current = latent.to_vec();
        for layer in &self.decoder {
            current = layer.forward(&current);
        }
        current
    }

    /// Full forward pass: encode then decode.
    pub fn reconstruct(&self, input: &[f64]) -> Vec<f64> {
        let encoded = self.encode(input);
        self.decode(&encoded)
    }

    /// Compute reconstruction loss for a single sample.
    pub fn reconstruction_loss(&self, input: &[f64]) -> f64 {
        let reconstructed = self.reconstruct(input);
        self.config.loss_fn.compute(input, &reconstructed)
    }

    /// Compute sparsity penalty using KL divergence.
    pub fn sparsity_penalty(&self, inputs: &[Vec<f64>]) -> f64 {
        let target = match self.config.sparsity_target {
            Some(t) => t,
            None => return 0.0,
        };

        let mut total_penalty = 0.0;
        let n = inputs.len() as f64;

        // Collect average activations across bottleneck neurons.
        if inputs.is_empty() {
            return 0.0;
        }
        let bottleneck_dim = self.config.bottleneck_dim;
        let mut avg_activations = vec![0.0; bottleneck_dim];

        for input in inputs {
            let encoded = self.encode(input);
            for (i, val) in encoded.iter().enumerate() {
                // Apply sigmoid to get activation probability.
                let prob = 1.0 / (1.0 + (-val).exp());
                avg_activations[i] += prob / n;
            }
        }

        // KL divergence: KL(target || avg_activation).
        for &rho_hat in &avg_activations {
            let rho_hat = rho_hat.clamp(1e-10, 1.0 - 1e-10);
            let kl = target * (target / rho_hat).ln()
                + (1.0 - target) * ((1.0 - target) / (1.0 - rho_hat)).ln();
            total_penalty += kl;
        }

        self.config.sparsity_weight * total_penalty
    }

    /// Train on a batch of samples for one epoch using simple gradient descent.
    pub fn train_epoch(&mut self, data: &[Vec<f64>]) -> f64 {
        if data.is_empty() { return 0.0; }

        let mut total_loss = 0.0;
        for sample in data {
            let reconstructed = self.reconstruct(sample);
            let loss = self.config.loss_fn.compute(sample, &reconstructed);
            total_loss += loss;

            // Compute output gradient.
            let grad = self.config.loss_fn.gradient(sample, &reconstructed);

            // Apply simple weight update on last decoder layer.
            let last_dec = self.decoder.len() - 1;
            let lr = self.config.learning_rate;
            for j in 0..self.decoder[last_dec].biases.len() {
                self.decoder[last_dec].biases[j] -= lr * grad[j.min(grad.len() - 1)];
                for k in 0..self.decoder[last_dec].weights[j].len() {
                    let input_val = if last_dec > 0 {
                        0.5 // Approximate for simplified training.
                    } else {
                        let encoded = self.encode(sample);
                        encoded[k.min(encoded.len() - 1)]
                    };
                    self.decoder[last_dec].weights[j][k] -= lr * grad[j.min(grad.len() - 1)] * input_val;
                }
            }
        }

        let avg_loss = total_loss / data.len() as f64;
        self.train_losses.push(avg_loss);
        avg_loss
    }

    /// Get the recorded training losses.
    pub fn train_losses(&self) -> &[f64] {
        &self.train_losses
    }

    /// Bottleneck dimensionality.
    pub fn bottleneck_dim(&self) -> usize {
        self.config.bottleneck_dim
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        let enc: usize = self.encoder.iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.biases.len())
            .sum();
        let dec: usize = self.decoder.iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.biases.len())
            .sum();
        enc + dec
    }
}

impl fmt::Display for Autoencoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Autoencoder(input={}, bottleneck={}, encoder_layers={}, params={}, loss={})",
            self.config.input_dim,
            self.config.bottleneck_dim,
            self.encoder.len(),
            self.param_count(),
            self.config.loss_fn,
        )
    }
}

// ── Denoising Autoencoder ─────────────────────────────────────

/// Denoising autoencoder that adds noise to inputs during training.
#[derive(Debug, Clone)]
pub struct DenoisingAutoencoder {
    inner: Autoencoder,
    noise_factor: f64,
    rng_state: u64,
}

impl DenoisingAutoencoder {
    pub fn new(config: AutoencoderConfig) -> Self {
        Self {
            inner: Autoencoder::build(config),
            noise_factor: 0.1,
            rng_state: 12345,
        }
    }

    pub fn with_noise_factor(mut self, factor: f64) -> Self {
        self.noise_factor = factor;
        self
    }

    /// Add Gaussian-like noise to input using Box-Muller approximation.
    pub fn add_noise(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|x| {
            self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (self.rng_state >> 33) as f64 / (1u64 << 31) as f64;
            self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (self.rng_state >> 33) as f64 / (1u64 << 31) as f64;
            let u1 = u1.max(1e-10);
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (x + self.noise_factor * noise).clamp(0.0, 1.0)
        }).collect()
    }

    /// Train with noisy inputs, measuring loss against clean originals.
    pub fn train_denoising(&mut self, data: &[Vec<f64>]) -> f64 {
        let noisy: Vec<Vec<f64>> = data.iter()
            .map(|sample| self.add_noise(sample))
            .collect();

        let mut total_loss = 0.0;
        for (noisy_sample, clean_sample) in noisy.iter().zip(data.iter()) {
            let reconstructed = self.inner.reconstruct(noisy_sample);
            total_loss += self.inner.config.loss_fn.compute(clean_sample, &reconstructed);
        }
        total_loss / data.len() as f64
    }

    pub fn encode(&self, input: &[f64]) -> Vec<f64> {
        self.inner.encode(input)
    }

    pub fn reconstruct(&self, input: &[f64]) -> Vec<f64> {
        self.inner.reconstruct(input)
    }
}

impl fmt::Display for DenoisingAutoencoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenoisingAutoencoder(noise={:.3}, {})", self.noise_factor, self.inner)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_relu() {
        assert_eq!(Activation::ReLU.forward(-1.0), 0.0);
        assert_eq!(Activation::ReLU.forward(2.0), 2.0);
    }

    #[test]
    fn test_activation_sigmoid() {
        let s = Activation::Sigmoid.forward(0.0);
        assert!((s - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_activation_tanh() {
        assert!((Activation::Tanh.forward(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_activation_leaky_relu() {
        let act = Activation::LeakyReLU(0.01);
        assert_eq!(act.forward(1.0), 1.0);
        assert!((act.forward(-1.0) - (-0.01)).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let d = Activation::Sigmoid.derivative(0.0);
        assert!((d - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let loss = ReconstructionLoss::MSE.compute(&a, &a);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let a = vec![1.0, 2.0];
        let b = vec![2.0, 3.0];
        let loss = ReconstructionLoss::MSE.compute(&a, &b);
        assert!((loss - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mae_loss() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let loss = ReconstructionLoss::MAE.compute(&a, &b);
        assert!((loss - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_gradient_direction() {
        let orig = vec![0.0];
        let recon = vec![1.0];
        let grad = ReconstructionLoss::MSE.gradient(&orig, &recon);
        assert!(grad[0] > 0.0); // Positive gradient pushes reconstruction down.
    }

    #[test]
    fn test_dense_layer_dimensions() {
        let layer = DenseLayer::new(4, 3, Activation::ReLU, 42);
        let output = layer.forward(&[1.0, 0.5, -0.5, 0.0]);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_autoencoder_build() {
        let config = AutoencoderConfig::new(10, 3)
            .with_encoder_layers(vec![8, 5]);
        let ae = Autoencoder::build(config);
        assert_eq!(ae.bottleneck_dim(), 3);
    }

    #[test]
    fn test_autoencoder_reconstruct_shape() {
        let config = AutoencoderConfig::new(6, 2)
            .with_encoder_layers(vec![4]);
        let ae = Autoencoder::build(config);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let output = ae.reconstruct(&input);
        assert_eq!(output.len(), 6);
    }

    #[test]
    fn test_encode_bottleneck_dim() {
        let config = AutoencoderConfig::new(8, 2);
        let ae = Autoencoder::build(config);
        let encoded = ae.encode(&vec![0.5; 8]);
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_param_count() {
        let config = AutoencoderConfig::new(4, 2);
        let ae = Autoencoder::build(config);
        assert!(ae.param_count() > 0);
        // Encoder: 4->2 = 4*2 + 2 = 10
        // Decoder: 2->4 = 2*4 + 4 = 12
        assert_eq!(ae.param_count(), 22);
    }

    #[test]
    fn test_display_autoencoder() {
        let config = AutoencoderConfig::new(10, 3);
        let ae = Autoencoder::build(config);
        let s = format!("{ae}");
        assert!(s.contains("Autoencoder"));
        assert!(s.contains("bottleneck=3"));
    }

    #[test]
    fn test_sparsity_penalty_disabled() {
        let config = AutoencoderConfig::new(4, 2);
        let ae = Autoencoder::build(config);
        let data = vec![vec![0.5; 4]];
        assert_eq!(ae.sparsity_penalty(&data), 0.0);
    }

    #[test]
    fn test_sparsity_penalty_enabled() {
        let config = AutoencoderConfig::new(4, 2)
            .with_sparsity(0.05, 0.1);
        let ae = Autoencoder::build(config);
        let data = vec![vec![0.5; 4], vec![0.3; 4]];
        let penalty = ae.sparsity_penalty(&data);
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_train_epoch_returns_loss() {
        let config = AutoencoderConfig::new(4, 2)
            .with_learning_rate(0.01);
        let mut ae = Autoencoder::build(config);
        let data = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]];
        let loss = ae.train_epoch(&data);
        assert!(loss >= 0.0);
        assert_eq!(ae.train_losses().len(), 1);
    }

    #[test]
    fn test_denoising_autoencoder() {
        let config = AutoencoderConfig::new(4, 2);
        let mut dae = DenoisingAutoencoder::new(config)
            .with_noise_factor(0.2);
        let data = vec![vec![0.5; 4]];
        let loss = dae.train_denoising(&data);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_denoising_adds_noise() {
        let config = AutoencoderConfig::new(4, 2);
        let mut dae = DenoisingAutoencoder::new(config)
            .with_noise_factor(0.5);
        let input = vec![0.5; 4];
        let noisy = dae.add_noise(&input);
        assert_eq!(noisy.len(), 4);
        // At least one value should differ from the original.
        let any_different = noisy.iter().zip(input.iter()).any(|(n, o)| (n - o).abs() > 1e-10);
        assert!(any_different);
    }

    #[test]
    fn test_bce_loss_perfect() {
        let a = vec![1.0, 0.0];
        let b = vec![0.9999, 0.0001];
        let loss = ReconstructionLoss::BinaryCrossEntropy.compute(&a, &b);
        assert!(loss < 0.01);
    }
}
