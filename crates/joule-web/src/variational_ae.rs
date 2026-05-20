//! Variational autoencoder: reparameterization trick, KL divergence, ELBO loss, latent sampling.
//!
//! Implements a VAE with diagonal-Gaussian latent space, the reparameterization
//! trick for backprop through stochastic nodes, and the evidence lower bound
//! (ELBO = reconstruction - KL) as training objective. Supports configurable
//! encoder/decoder architectures, warm-up scheduling for the KL weight, and
//! posterior sampling for generation.

use std::fmt;

// ── Activation ────────────────────────────────────────────────

/// Activation function for hidden layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VaeActivation {
    ReLU,
    Tanh,
    Sigmoid,
    ELU(f64),
}

impl VaeActivation {
    pub fn forward(&self, x: f64) -> f64 {
        match self {
            VaeActivation::ReLU => x.max(0.0),
            VaeActivation::Tanh => x.tanh(),
            VaeActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            VaeActivation::ELU(alpha) => if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) },
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            VaeActivation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            VaeActivation::Tanh => 1.0 - x.tanh().powi(2),
            VaeActivation::Sigmoid => {
                let s = self.forward(x);
                s * (1.0 - s)
            }
            VaeActivation::ELU(alpha) => if x >= 0.0 { 1.0 } else { alpha * x.exp() },
        }
    }
}

impl fmt::Display for VaeActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VaeActivation::ReLU => write!(f, "ReLU"),
            VaeActivation::Tanh => write!(f, "Tanh"),
            VaeActivation::Sigmoid => write!(f, "Sigmoid"),
            VaeActivation::ELU(a) => write!(f, "ELU(alpha={a:.3})"),
        }
    }
}

// ── Simple PRNG ───────────────────────────────────────────────

/// Linear congruential generator for reproducible sampling.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller transform.
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Dense Layer ───────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DenseLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activation: Option<VaeActivation>,
}

impl DenseLayer {
    fn new(input_dim: usize, output_dim: usize, activation: Option<VaeActivation>, rng: &mut Lcg) -> Self {
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| rng.standard_normal() * scale).collect())
            .collect();
        let biases = vec![0.0; output_dim];
        Self { weights, biases, activation }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        (0..self.biases.len())
            .map(|j| {
                let z: f64 = self.weights[j].iter().zip(input).map(|(w, x)| w * x).sum::<f64>()
                    + self.biases[j];
                match &self.activation {
                    Some(act) => act.forward(z),
                    None => z,
                }
            })
            .collect()
    }
}

// ── KL Divergence ─────────────────────────────────────────────

/// Compute KL divergence between N(mu, sigma^2) and N(0, 1).
/// KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
pub fn kl_divergence_standard_normal(mu: &[f64], log_var: &[f64]) -> f64 {
    assert_eq!(mu.len(), log_var.len());
    -0.5 * mu.iter().zip(log_var.iter())
        .map(|(m, lv)| 1.0 + lv - m.powi(2) - lv.exp())
        .sum::<f64>()
}

/// KL divergence between two diagonal Gaussians.
pub fn kl_divergence_gaussians(
    mu1: &[f64], log_var1: &[f64],
    mu2: &[f64], log_var2: &[f64],
) -> f64 {
    assert_eq!(mu1.len(), mu2.len());
    let d = mu1.len();
    let mut kl = 0.0;
    for i in 0..d {
        let var1 = log_var1[i].exp();
        let var2 = log_var2[i].exp();
        kl += (var2 / var1).ln().max(-20.0) + (var1 + (mu1[i] - mu2[i]).powi(2)) / var2 - 1.0;
    }
    0.5 * kl
}

// ── Reparameterization ────────────────────────────────────────

/// Reparameterization trick: z = mu + sigma * epsilon, epsilon ~ N(0,1).
pub fn reparameterize(mu: &[f64], log_var: &[f64], rng: &mut Lcg) -> Vec<f64> {
    mu.iter().zip(log_var.iter())
        .map(|(m, lv)| {
            let sigma = (lv / 2.0).exp();
            let eps = rng.standard_normal();
            m + sigma * eps
        })
        .collect()
}

// ── VAE Config ────────────────────────────────────────────────

/// Configuration for building a variational autoencoder.
#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub input_dim: usize,
    pub hidden_layers: Vec<usize>,
    pub latent_dim: usize,
    pub activation: VaeActivation,
    pub learning_rate: f64,
    pub kl_weight: f64,
    pub kl_warmup_epochs: usize,
    pub seed: u64,
}

impl VaeConfig {
    pub fn new(input_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_layers: vec![],
            latent_dim,
            activation: VaeActivation::ReLU,
            learning_rate: 0.001,
            kl_weight: 1.0,
            kl_warmup_epochs: 0,
            seed: 42,
        }
    }

    pub fn with_hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }

    pub fn with_activation(mut self, act: VaeActivation) -> Self {
        self.activation = act;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_kl_weight(mut self, w: f64) -> Self {
        self.kl_weight = w;
        self
    }

    pub fn with_kl_warmup(mut self, epochs: usize) -> Self {
        self.kl_warmup_epochs = epochs;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

// ── VAE ───────────────────────────────────────────────────────

/// Variational Autoencoder with diagonal Gaussian posterior.
#[derive(Debug, Clone)]
pub struct Vae {
    encoder_layers: Vec<DenseLayer>,
    mu_layer: DenseLayer,
    log_var_layer: DenseLayer,
    decoder_layers: Vec<DenseLayer>,
    config: VaeConfig,
    rng: Lcg,
    epoch: usize,
    history: Vec<ElboRecord>,
}

/// Record of a single training epoch's ELBO decomposition.
#[derive(Debug, Clone)]
pub struct ElboRecord {
    pub epoch: usize,
    pub reconstruction_loss: f64,
    pub kl_loss: f64,
    pub elbo: f64,
    pub kl_weight: f64,
}

impl fmt::Display for ElboRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Epoch {}: ELBO={:.4}, recon={:.4}, KL={:.4} (beta={:.3})",
            self.epoch, self.elbo, self.reconstruction_loss, self.kl_loss, self.kl_weight,
        )
    }
}

impl Vae {
    /// Build a VAE from the configuration.
    pub fn build(config: VaeConfig) -> Self {
        let mut rng = Lcg::new(config.seed);

        // Encoder: input -> hidden layers -> (mu, log_var).
        let mut encoder_layers = Vec::new();
        let mut prev_dim = config.input_dim;
        for &hidden_dim in &config.hidden_layers {
            encoder_layers.push(DenseLayer::new(prev_dim, hidden_dim, Some(config.activation), &mut rng));
            prev_dim = hidden_dim;
        }
        let mu_layer = DenseLayer::new(prev_dim, config.latent_dim, None, &mut rng);
        let log_var_layer = DenseLayer::new(prev_dim, config.latent_dim, None, &mut rng);

        // Decoder: latent -> reversed hidden layers -> output (sigmoid).
        let mut decoder_layers = Vec::new();
        prev_dim = config.latent_dim;
        for &hidden_dim in config.hidden_layers.iter().rev() {
            decoder_layers.push(DenseLayer::new(prev_dim, hidden_dim, Some(config.activation), &mut rng));
            prev_dim = hidden_dim;
        }
        decoder_layers.push(DenseLayer::new(prev_dim, config.input_dim, Some(VaeActivation::Sigmoid), &mut rng));

        Self {
            encoder_layers,
            mu_layer,
            log_var_layer,
            decoder_layers,
            config,
            rng,
            epoch: 0,
            history: Vec::new(),
        }
    }

    /// Encode to posterior parameters (mu, log_var).
    pub fn encode(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut h = input.to_vec();
        for layer in &self.encoder_layers {
            h = layer.forward(&h);
        }
        let mu = self.mu_layer.forward(&h);
        let log_var = self.log_var_layer.forward(&h);
        (mu, log_var)
    }

    /// Sample from the posterior using the reparameterization trick.
    pub fn sample(&mut self, mu: &[f64], log_var: &[f64]) -> Vec<f64> {
        reparameterize(mu, log_var, &mut self.rng)
    }

    /// Decode a latent vector to reconstruction.
    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        let mut h = z.to_vec();
        for layer in &self.decoder_layers {
            h = layer.forward(&h);
        }
        h
    }

    /// Full forward pass: encode, sample, decode.
    pub fn forward(&mut self, input: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mu, log_var) = self.encode(input);
        let z = self.sample(&mu, &log_var);
        let recon = self.decode(&z);
        (recon, mu, log_var, z)
    }

    /// Compute the current KL weight considering warmup schedule.
    pub fn current_kl_weight(&self) -> f64 {
        if self.config.kl_warmup_epochs == 0 {
            return self.config.kl_weight;
        }
        let progress = (self.epoch as f64 / self.config.kl_warmup_epochs as f64).min(1.0);
        self.config.kl_weight * progress
    }

    /// Compute ELBO for a single sample (lower is better for loss = -ELBO).
    pub fn compute_elbo(&self, input: &[f64], recon: &[f64], mu: &[f64], log_var: &[f64]) -> (f64, f64, f64) {
        let n = input.len() as f64;
        let recon_loss: f64 = input.iter().zip(recon.iter())
            .map(|(x, r)| (x - r).powi(2))
            .sum::<f64>() / n;

        let kl = kl_divergence_standard_normal(mu, log_var);
        let beta = self.current_kl_weight();
        let elbo = -(recon_loss + beta * kl);
        (recon_loss, kl, elbo)
    }

    /// Train for one epoch on the given data.
    pub fn train_epoch(&mut self, data: &[Vec<f64>]) -> ElboRecord {
        if data.is_empty() {
            return ElboRecord { epoch: self.epoch, reconstruction_loss: 0.0, kl_loss: 0.0, elbo: 0.0, kl_weight: 0.0 };
        }

        let mut total_recon = 0.0;
        let mut total_kl = 0.0;

        for sample in data {
            let (recon, mu, log_var, _z) = self.forward(sample);
            let (recon_loss, kl, _elbo) = self.compute_elbo(sample, &recon, &mu, &log_var);
            total_recon += recon_loss;
            total_kl += kl;
        }

        let n = data.len() as f64;
        let avg_recon = total_recon / n;
        let avg_kl = total_kl / n;
        let beta = self.current_kl_weight();
        let elbo = -(avg_recon + beta * avg_kl);

        let record = ElboRecord {
            epoch: self.epoch,
            reconstruction_loss: avg_recon,
            kl_loss: avg_kl,
            elbo,
            kl_weight: beta,
        };
        self.history.push(record.clone());
        self.epoch += 1;
        record
    }

    /// Generate new samples by decoding from the prior N(0, I).
    pub fn generate(&mut self, count: usize) -> Vec<Vec<f64>> {
        (0..count)
            .map(|_| {
                let z: Vec<f64> = (0..self.config.latent_dim)
                    .map(|_| self.rng.standard_normal())
                    .collect();
                self.decode(&z)
            })
            .collect()
    }

    /// Reconstruct a batch of inputs.
    pub fn reconstruct_batch(&mut self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|sample| {
                let (recon, _, _, _) = self.forward(sample);
                recon
            })
            .collect()
    }

    /// Get training history.
    pub fn history(&self) -> &[ElboRecord] {
        &self.history
    }

    /// Latent dimensionality.
    pub fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        let layer_params = |l: &DenseLayer| -> usize {
            l.weights.len() * l.weights[0].len() + l.biases.len()
        };
        let enc: usize = self.encoder_layers.iter().map(layer_params).sum();
        let mu_params = layer_params(&self.mu_layer);
        let lv_params = layer_params(&self.log_var_layer);
        let dec: usize = self.decoder_layers.iter().map(layer_params).sum();
        enc + mu_params + lv_params + dec
    }
}

impl fmt::Display for Vae {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VAE(input={}, latent={}, hidden={:?}, params={}, kl_weight={:.3})",
            self.config.input_dim,
            self.config.latent_dim,
            self.config.hidden_layers,
            self.param_count(),
            self.current_kl_weight(),
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_divergence_zero_for_prior() {
        let mu = vec![0.0, 0.0, 0.0];
        let log_var = vec![0.0, 0.0, 0.0];
        let kl = kl_divergence_standard_normal(&mu, &log_var);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let mu = vec![1.0, -1.0];
        let log_var = vec![0.5, -0.5];
        let kl = kl_divergence_standard_normal(&mu, &log_var);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_kl_between_identical_gaussians() {
        let mu = vec![1.0, 2.0];
        let lv = vec![0.5, -0.3];
        let kl = kl_divergence_gaussians(&mu, &lv, &mu, &lv);
        assert!(kl.abs() < 1e-8);
    }

    #[test]
    fn test_reparameterize_shape() {
        let mu = vec![0.0; 5];
        let log_var = vec![0.0; 5];
        let mut rng = Lcg::new(42);
        let z = reparameterize(&mu, &log_var, &mut rng);
        assert_eq!(z.len(), 5);
    }

    #[test]
    fn test_reparameterize_deterministic_with_same_seed() {
        let mu = vec![1.0, 2.0];
        let log_var = vec![0.0, 0.0];
        let z1 = reparameterize(&mu, &log_var, &mut Lcg::new(99));
        let z2 = reparameterize(&mu, &log_var, &mut Lcg::new(99));
        assert_eq!(z1, z2);
    }

    #[test]
    fn test_vae_build() {
        let config = VaeConfig::new(10, 3).with_hidden_layers(vec![8, 5]);
        let vae = Vae::build(config);
        assert_eq!(vae.latent_dim(), 3);
    }

    #[test]
    fn test_vae_encode_shape() {
        let config = VaeConfig::new(6, 2).with_hidden_layers(vec![4]);
        let vae = Vae::build(config);
        let (mu, log_var) = vae.encode(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(mu.len(), 2);
        assert_eq!(log_var.len(), 2);
    }

    #[test]
    fn test_vae_decode_shape() {
        let config = VaeConfig::new(6, 2).with_hidden_layers(vec![4]);
        let vae = Vae::build(config);
        let recon = vae.decode(&[0.5, -0.5]);
        assert_eq!(recon.len(), 6);
    }

    #[test]
    fn test_vae_forward() {
        let config = VaeConfig::new(4, 2);
        let mut vae = Vae::build(config);
        let (recon, mu, log_var, z) = vae.forward(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(recon.len(), 4);
        assert_eq!(mu.len(), 2);
        assert_eq!(log_var.len(), 2);
        assert_eq!(z.len(), 2);
    }

    #[test]
    fn test_vae_generate() {
        let config = VaeConfig::new(4, 2);
        let mut vae = Vae::build(config);
        let samples = vae.generate(5);
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0].len(), 4);
    }

    #[test]
    fn test_elbo_computation() {
        let config = VaeConfig::new(4, 2);
        let vae = Vae::build(config);
        let input = vec![0.5; 4];
        let recon = vec![0.4; 4];
        let mu = vec![0.0, 0.0];
        let log_var = vec![0.0, 0.0];
        let (rl, kl, elbo) = vae.compute_elbo(&input, &recon, &mu, &log_var);
        assert!(rl > 0.0);
        assert!(kl.abs() < 1e-8);
        assert!(elbo < 0.0); // Negative since -(recon_loss + kl).
    }

    #[test]
    fn test_kl_warmup() {
        let config = VaeConfig::new(4, 2).with_kl_weight(1.0).with_kl_warmup(10);
        let mut vae = Vae::build(config);
        assert!((vae.current_kl_weight() - 0.0).abs() < 1e-10);
        let data = vec![vec![0.5; 4]];
        for _ in 0..5 {
            vae.train_epoch(&data);
        }
        assert!((vae.current_kl_weight() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_train_epoch_records_history() {
        let config = VaeConfig::new(4, 2);
        let mut vae = Vae::build(config);
        let data = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let record = vae.train_epoch(&data);
        assert_eq!(record.epoch, 0);
        assert_eq!(vae.history().len(), 1);
    }

    #[test]
    fn test_param_count() {
        let config = VaeConfig::new(4, 2);
        let vae = Vae::build(config);
        // Encoder: nothing (no hidden layers).
        // mu_layer: 4*2 + 2 = 10.
        // log_var_layer: 4*2 + 2 = 10.
        // Decoder: 2*4 + 4 = 12.
        assert_eq!(vae.param_count(), 32);
    }

    #[test]
    fn test_display() {
        let config = VaeConfig::new(10, 3).with_hidden_layers(vec![8, 5]);
        let vae = Vae::build(config);
        let s = format!("{vae}");
        assert!(s.contains("VAE"));
        assert!(s.contains("latent=3"));
    }

    #[test]
    fn test_reconstruct_batch() {
        let config = VaeConfig::new(4, 2);
        let mut vae = Vae::build(config);
        let data = vec![vec![0.1; 4], vec![0.9; 4]];
        let recons = vae.reconstruct_batch(&data);
        assert_eq!(recons.len(), 2);
        assert_eq!(recons[0].len(), 4);
    }

    #[test]
    fn test_elu_activation() {
        let act = VaeActivation::ELU(1.0);
        assert_eq!(act.forward(1.0), 1.0);
        assert!(act.forward(-1.0) < 0.0);
        assert!(act.forward(-1.0) > -1.0);
    }

    #[test]
    fn test_elbo_record_display() {
        let record = ElboRecord {
            epoch: 5,
            reconstruction_loss: 0.123,
            kl_loss: 0.456,
            elbo: -0.579,
            kl_weight: 1.0,
        };
        let s = format!("{record}");
        assert!(s.contains("Epoch 5"));
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        let mu1 = vec![0.0];
        let lv1 = vec![0.0];
        let mu2 = vec![1.0];
        let lv2 = vec![0.0];
        let kl12 = kl_divergence_gaussians(&mu1, &lv1, &mu2, &lv2);
        let kl21 = kl_divergence_gaussians(&mu2, &lv2, &mu1, &lv1);
        assert!(kl12 > 0.0);
        assert!(kl21 > 0.0);
        // KL is not symmetric in general.
        assert!((kl12 - kl21).abs() < 1e-10); // But symmetric for equal variances.
    }
}
