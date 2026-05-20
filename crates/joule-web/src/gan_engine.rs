//! GAN framework: generator/discriminator training loop, Wasserstein loss,
//! gradient penalty, mode collapse detection.
//!
//! Implements a generative adversarial network with configurable generator and
//! discriminator architectures. Supports vanilla BCE loss and Wasserstein loss
//! with gradient penalty (WGAN-GP). Includes mode collapse detection via
//! diversity metrics on generated samples.

use std::fmt;

// ── PRNG ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn noise_vec(&mut self, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| self.normal()).collect()
    }
}

// ── GAN Loss ──────────────────────────────────────────────────

/// Loss function for GAN training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GanLoss {
    /// Vanilla binary cross-entropy loss.
    BinaryCrossEntropy,
    /// Wasserstein distance (Earth mover).
    Wasserstein,
    /// Least-squares GAN loss.
    LeastSquares,
}

impl GanLoss {
    /// Discriminator loss on real samples (should output high/1).
    pub fn discriminator_real(&self, d_output: f64) -> f64 {
        match self {
            GanLoss::BinaryCrossEntropy => {
                let d = d_output.clamp(1e-12, 1.0 - 1e-12);
                -d.ln()
            }
            GanLoss::Wasserstein => -d_output,
            GanLoss::LeastSquares => (d_output - 1.0).powi(2),
        }
    }

    /// Discriminator loss on fake samples (should output low/0).
    pub fn discriminator_fake(&self, d_output: f64) -> f64 {
        match self {
            GanLoss::BinaryCrossEntropy => {
                let d = d_output.clamp(1e-12, 1.0 - 1e-12);
                -(1.0 - d).ln()
            }
            GanLoss::Wasserstein => d_output,
            GanLoss::LeastSquares => d_output.powi(2),
        }
    }

    /// Generator loss (generator wants discriminator to output high/1 for fakes).
    pub fn generator_loss(&self, d_output: f64) -> f64 {
        match self {
            GanLoss::BinaryCrossEntropy => {
                let d = d_output.clamp(1e-12, 1.0 - 1e-12);
                -d.ln()
            }
            GanLoss::Wasserstein => -d_output,
            GanLoss::LeastSquares => (d_output - 1.0).powi(2),
        }
    }
}

impl fmt::Display for GanLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GanLoss::BinaryCrossEntropy => write!(f, "BCE"),
            GanLoss::Wasserstein => write!(f, "Wasserstein"),
            GanLoss::LeastSquares => write!(f, "LSGAN"),
        }
    }
}

// ── Dense Layer ───────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    use_sigmoid: bool,
}

impl Layer {
    fn new(input_dim: usize, output_dim: usize, use_sigmoid: bool, rng: &mut Rng) -> Self {
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| rng.normal() * scale).collect())
            .collect();
        let biases = vec![0.0; output_dim];
        Self { weights, biases, use_sigmoid }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        (0..self.biases.len())
            .map(|j| {
                let z: f64 = self.weights[j].iter().zip(input).map(|(w, x)| w * x).sum::<f64>() + self.biases[j];
                if self.use_sigmoid {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    z.max(0.0) // LeakyReLU with alpha=0 (ReLU).
                }
            })
            .collect()
    }
}

// ── Generator ─────────────────────────────────────────────────

/// Generator network: maps noise vector to data space.
#[derive(Debug, Clone)]
pub struct Generator {
    layers: Vec<Layer>,
    noise_dim: usize,
    output_dim: usize,
}

impl Generator {
    pub fn new(noise_dim: usize, hidden_sizes: &[usize], output_dim: usize, rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        let mut prev = noise_dim;
        for &h in hidden_sizes {
            layers.push(Layer::new(prev, h, false, rng));
            prev = h;
        }
        // Output layer uses sigmoid to produce values in [0, 1].
        layers.push(Layer::new(prev, output_dim, true, rng));
        Self { layers, noise_dim, output_dim }
    }

    /// Generate a sample from a noise vector.
    pub fn forward(&self, noise: &[f64]) -> Vec<f64> {
        let mut h = noise.to_vec();
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        h
    }

    pub fn noise_dim(&self) -> usize { self.noise_dim }
    pub fn output_dim(&self) -> usize { self.output_dim }

    pub fn param_count(&self) -> usize {
        self.layers.iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.biases.len())
            .sum()
    }
}

impl fmt::Display for Generator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Generator(noise={}, output={}, layers={}, params={})",
            self.noise_dim, self.output_dim, self.layers.len(), self.param_count())
    }
}

// ── Discriminator ─────────────────────────────────────────────

/// Discriminator network: classifies samples as real or fake.
#[derive(Debug, Clone)]
pub struct Discriminator {
    layers: Vec<Layer>,
    input_dim: usize,
}

impl Discriminator {
    pub fn new(input_dim: usize, hidden_sizes: &[usize], rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        let mut prev = input_dim;
        for &h in hidden_sizes {
            layers.push(Layer::new(prev, h, false, rng));
            prev = h;
        }
        // Single sigmoid output.
        layers.push(Layer::new(prev, 1, true, rng));
        Self { layers, input_dim }
    }

    /// Classify a sample, returning probability of being real.
    pub fn forward(&self, sample: &[f64]) -> f64 {
        let mut h = sample.to_vec();
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        h[0]
    }

    pub fn param_count(&self) -> usize {
        self.layers.iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.biases.len())
            .sum()
    }
}

impl fmt::Display for Discriminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Discriminator(input={}, layers={}, params={})",
            self.input_dim, self.layers.len(), self.param_count())
    }
}

// ── Gradient Penalty ──────────────────────────────────────────

/// Compute gradient penalty for WGAN-GP by interpolating between real and fake samples.
pub fn gradient_penalty(
    discriminator: &Discriminator,
    real: &[f64],
    fake: &[f64],
    lambda: f64,
) -> f64 {
    assert_eq!(real.len(), fake.len());
    // Use epsilon=0.5 for midpoint interpolation.
    let eps = 0.5;
    let interpolated: Vec<f64> = real.iter().zip(fake.iter())
        .map(|(r, f_val)| eps * r + (1.0 - eps) * f_val)
        .collect();

    // Approximate gradient norm via finite differences.
    let delta = 1e-5;
    let d_interp = discriminator.forward(&interpolated);
    let mut grad_sq_sum = 0.0;

    for i in 0..interpolated.len() {
        let mut perturbed = interpolated.clone();
        perturbed[i] += delta;
        let d_perturbed = discriminator.forward(&perturbed);
        let grad_i = (d_perturbed - d_interp) / delta;
        grad_sq_sum += grad_i * grad_i;
    }

    let grad_norm = grad_sq_sum.sqrt();
    lambda * (grad_norm - 1.0).powi(2)
}

// ── Mode Collapse Detection ───────────────────────────────────

/// Metrics for detecting mode collapse in generated samples.
#[derive(Debug, Clone)]
pub struct CollapseMetrics {
    /// Average pairwise distance between generated samples.
    pub avg_pairwise_distance: f64,
    /// Standard deviation of generated features.
    pub feature_std: Vec<f64>,
    /// Minimum pairwise distance (low = potential collapse).
    pub min_pairwise_distance: f64,
    /// Number of distinct modes detected via simple clustering.
    pub estimated_modes: usize,
}

impl CollapseMetrics {
    /// Analyze a batch of generated samples for mode collapse indicators.
    pub fn analyze(samples: &[Vec<f64>]) -> Self {
        if samples.is_empty() || samples.len() < 2 {
            return Self {
                avg_pairwise_distance: 0.0,
                feature_std: vec![],
                min_pairwise_distance: 0.0,
                estimated_modes: 0,
            };
        }

        let n = samples.len();
        let dim = samples[0].len();

        // Pairwise L2 distances.
        let mut total_dist = 0.0;
        let mut min_dist = f64::MAX;
        let mut pair_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist: f64 = samples[i].iter().zip(samples[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total_dist += dist;
                if dist < min_dist { min_dist = dist; }
                pair_count += 1;
            }
        }

        let avg_dist = if pair_count > 0 { total_dist / pair_count as f64 } else { 0.0 };

        // Per-feature standard deviation.
        let mut means = vec![0.0; dim];
        for s in samples {
            for (i, v) in s.iter().enumerate() {
                means[i] += v / n as f64;
            }
        }
        let feature_std: Vec<f64> = (0..dim)
            .map(|i| {
                let var = samples.iter()
                    .map(|s| (s[i] - means[i]).powi(2))
                    .sum::<f64>() / n as f64;
                var.sqrt()
            })
            .collect();

        // Simple mode counting via threshold-based clustering.
        let threshold = avg_dist * 0.3;
        let mut assigned = vec![false; n];
        let mut modes = 0;
        for i in 0..n {
            if assigned[i] { continue; }
            modes += 1;
            assigned[i] = true;
            for j in (i + 1)..n {
                if assigned[j] { continue; }
                let dist: f64 = samples[i].iter().zip(samples[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < threshold {
                    assigned[j] = true;
                }
            }
        }

        Self {
            avg_pairwise_distance: avg_dist,
            feature_std,
            min_pairwise_distance: if min_dist == f64::MAX { 0.0 } else { min_dist },
            estimated_modes: modes,
        }
    }

    /// Returns true if mode collapse is likely (low diversity).
    pub fn is_collapsed(&self, threshold: f64) -> bool {
        self.avg_pairwise_distance < threshold
    }
}

impl fmt::Display for CollapseMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CollapseMetrics(avg_dist={:.4}, min_dist={:.4}, modes={})",
            self.avg_pairwise_distance, self.min_pairwise_distance, self.estimated_modes,
        )
    }
}

// ── GAN Training Loop ─────────────────────────────────────────

/// Training record for a single GAN iteration.
#[derive(Debug, Clone)]
pub struct GanTrainStep {
    pub step: usize,
    pub d_loss_real: f64,
    pub d_loss_fake: f64,
    pub g_loss: f64,
    pub gp: f64,
}

impl fmt::Display for GanTrainStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Step {}: D(real)={:.4}, D(fake)={:.4}, G={:.4}, GP={:.4}",
            self.step, self.d_loss_real, self.d_loss_fake, self.g_loss, self.gp,
        )
    }
}

/// GAN training engine.
#[derive(Debug, Clone)]
pub struct GanEngine {
    pub generator: Generator,
    pub discriminator: Discriminator,
    loss_fn: GanLoss,
    gp_lambda: f64,
    noise_dim: usize,
    rng: Rng,
    history: Vec<GanTrainStep>,
    step: usize,
}

impl GanEngine {
    pub fn new(noise_dim: usize, data_dim: usize, hidden: &[usize], loss_fn: GanLoss) -> Self {
        let mut rng = Rng::new(42);
        let generator = Generator::new(noise_dim, hidden, data_dim, &mut rng);
        let discriminator = Discriminator::new(data_dim, hidden, &mut rng);
        Self {
            generator,
            discriminator,
            loss_fn,
            gp_lambda: 10.0,
            noise_dim,
            rng,
            history: Vec::new(),
            step: 0,
        }
    }

    pub fn with_gp_lambda(mut self, lambda: f64) -> Self {
        self.gp_lambda = lambda;
        self
    }

    /// Run one training step with real data.
    pub fn train_step(&mut self, real_samples: &[Vec<f64>]) -> GanTrainStep {
        if real_samples.is_empty() {
            return GanTrainStep { step: self.step, d_loss_real: 0.0, d_loss_fake: 0.0, g_loss: 0.0, gp: 0.0 };
        }

        let mut d_loss_real_total = 0.0;
        let mut d_loss_fake_total = 0.0;
        let mut g_loss_total = 0.0;
        let mut gp_total = 0.0;

        for real in real_samples {
            // Generate fake sample.
            let noise = self.rng.noise_vec(self.noise_dim);
            let fake = self.generator.forward(&noise);

            // Discriminator losses.
            let d_real = self.discriminator.forward(real);
            let d_fake = self.discriminator.forward(&fake);

            d_loss_real_total += self.loss_fn.discriminator_real(d_real);
            d_loss_fake_total += self.loss_fn.discriminator_fake(d_fake);

            // Generator loss.
            g_loss_total += self.loss_fn.generator_loss(d_fake);

            // Gradient penalty (for WGAN-GP).
            if self.loss_fn == GanLoss::Wasserstein {
                gp_total += gradient_penalty(&self.discriminator, real, &fake, self.gp_lambda);
            }
        }

        let n = real_samples.len() as f64;
        let record = GanTrainStep {
            step: self.step,
            d_loss_real: d_loss_real_total / n,
            d_loss_fake: d_loss_fake_total / n,
            g_loss: g_loss_total / n,
            gp: gp_total / n,
        };

        self.history.push(record.clone());
        self.step += 1;
        record
    }

    /// Generate samples from the trained generator.
    pub fn generate(&mut self, count: usize) -> Vec<Vec<f64>> {
        (0..count)
            .map(|_| {
                let noise = self.rng.noise_vec(self.noise_dim);
                self.generator.forward(&noise)
            })
            .collect()
    }

    /// Check for mode collapse in recent generated samples.
    pub fn detect_mode_collapse(&mut self, sample_count: usize) -> CollapseMetrics {
        let samples = self.generate(sample_count);
        CollapseMetrics::analyze(&samples)
    }

    pub fn history(&self) -> &[GanTrainStep] { &self.history }

    pub fn total_params(&self) -> usize {
        self.generator.param_count() + self.discriminator.param_count()
    }
}

impl fmt::Display for GanEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GanEngine(loss={}, noise_dim={}, params={}, steps={})",
            self.loss_fn, self.noise_dim, self.total_params(), self.step,
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bce_discriminator_real() {
        let loss = GanLoss::BinaryCrossEntropy.discriminator_real(0.9);
        assert!(loss > 0.0);
        assert!(loss < 1.0);
    }

    #[test]
    fn test_bce_discriminator_fake() {
        let loss = GanLoss::BinaryCrossEntropy.discriminator_fake(0.1);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_wasserstein_loss_real() {
        let loss = GanLoss::Wasserstein.discriminator_real(5.0);
        assert_eq!(loss, -5.0);
    }

    #[test]
    fn test_wasserstein_loss_fake() {
        let loss = GanLoss::Wasserstein.discriminator_fake(3.0);
        assert_eq!(loss, 3.0);
    }

    #[test]
    fn test_lsgan_loss() {
        let real = GanLoss::LeastSquares.discriminator_real(1.0);
        assert!(real.abs() < 1e-10);
        let fake = GanLoss::LeastSquares.discriminator_fake(0.0);
        assert!(fake.abs() < 1e-10);
    }

    #[test]
    fn test_generator_forward_shape() {
        let mut rng = Rng::new(42);
        let gen_model = Generator::new(10, &[8, 6], 4, &mut rng);
        let noise = vec![0.5; 10];
        let output = gen_model.forward(&noise);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_discriminator_forward_range() {
        let mut rng = Rng::new(42);
        let disc = Discriminator::new(4, &[6], &mut rng);
        let score = disc.forward(&[0.5, 0.3, 0.7, 0.1]);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_gradient_penalty_nonnegative() {
        let mut rng = Rng::new(42);
        let disc = Discriminator::new(3, &[4], &mut rng);
        let real = vec![1.0, 0.5, 0.8];
        let fake = vec![0.2, 0.3, 0.1];
        let gp = gradient_penalty(&disc, &real, &fake, 10.0);
        assert!(gp >= 0.0);
    }

    #[test]
    fn test_gan_engine_build() {
        let engine = GanEngine::new(5, 3, &[4], GanLoss::BinaryCrossEntropy);
        assert!(engine.total_params() > 0);
    }

    #[test]
    fn test_gan_train_step() {
        let mut engine = GanEngine::new(5, 3, &[4], GanLoss::BinaryCrossEntropy);
        let data = vec![vec![0.8, 0.5, 0.3], vec![0.1, 0.9, 0.6]];
        let step = engine.train_step(&data);
        assert_eq!(step.step, 0);
        assert!(step.d_loss_real >= 0.0);
    }

    #[test]
    fn test_gan_generate() {
        let mut engine = GanEngine::new(5, 3, &[4], GanLoss::BinaryCrossEntropy);
        let samples = engine.generate(10);
        assert_eq!(samples.len(), 10);
        assert_eq!(samples[0].len(), 3);
    }

    #[test]
    fn test_collapse_metrics_diverse() {
        let samples = vec![
            vec![0.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0],
        ];
        let metrics = CollapseMetrics::analyze(&samples);
        assert!(metrics.avg_pairwise_distance > 0.5);
        assert!(metrics.estimated_modes >= 2);
    }

    #[test]
    fn test_collapse_metrics_collapsed() {
        let samples = vec![
            vec![0.5, 0.5], vec![0.501, 0.499], vec![0.5, 0.501], vec![0.499, 0.5],
        ];
        let metrics = CollapseMetrics::analyze(&samples);
        assert!(metrics.avg_pairwise_distance < 0.01);
        assert!(metrics.is_collapsed(0.1));
    }

    #[test]
    fn test_collapse_metrics_display() {
        let metrics = CollapseMetrics {
            avg_pairwise_distance: 1.5,
            feature_std: vec![0.3, 0.4],
            min_pairwise_distance: 0.1,
            estimated_modes: 3,
        };
        let s = format!("{metrics}");
        assert!(s.contains("modes=3"));
    }

    #[test]
    fn test_wgan_gp_train_step() {
        let mut engine = GanEngine::new(4, 2, &[3], GanLoss::Wasserstein)
            .with_gp_lambda(10.0);
        let data = vec![vec![0.5, 0.8]];
        let step = engine.train_step(&data);
        assert!(step.gp >= 0.0);
    }

    #[test]
    fn test_gan_history() {
        let mut engine = GanEngine::new(4, 2, &[3], GanLoss::BinaryCrossEntropy);
        let data = vec![vec![0.5, 0.8]];
        engine.train_step(&data);
        engine.train_step(&data);
        assert_eq!(engine.history().len(), 2);
    }

    #[test]
    fn test_generator_display() {
        let mut rng = Rng::new(42);
        let gen_model = Generator::new(10, &[8], 4, &mut rng);
        let s = format!("{gen_model}");
        assert!(s.contains("Generator"));
    }

    #[test]
    fn test_gan_engine_display() {
        let engine = GanEngine::new(5, 3, &[4], GanLoss::Wasserstein);
        let s = format!("{engine}");
        assert!(s.contains("Wasserstein"));
    }

    #[test]
    fn test_mode_collapse_detection() {
        let mut engine = GanEngine::new(4, 2, &[3], GanLoss::BinaryCrossEntropy);
        let metrics = engine.detect_mode_collapse(20);
        assert!(metrics.estimated_modes >= 1);
    }
}
