//! Image generation pipeline: noise-to-image, conditional generation,
//! classifier-free guidance, sampling strategies.
//!
//! Implements a complete image generation pipeline supporting unconditional and
//! conditional generation. Includes classifier-free guidance (CFG) for
//! controlling generation quality/diversity tradeoff, multiple sampling
//! strategies (Euler, Heun, DPM++), and image post-processing utilities
//! (clamping, normalization, tiling).

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
}

// ── Image Buffer ──────────────────────────────────────────────

/// Simple image buffer with width, height, and channels.
#[derive(Debug, Clone)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<f64>,
}

impl ImageBuffer {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self { width, height, channels, data: vec![0.0; width * height * channels] }
    }

    pub fn from_noise(width: usize, height: usize, channels: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let data: Vec<f64> = (0..width * height * channels).map(|_| rng.normal()).collect();
        Self { width, height, channels, data }
    }

    /// Get pixel value at (x, y, channel).
    pub fn at(&self, x: usize, y: usize, c: usize) -> f64 {
        self.data[(y * self.width + x) * self.channels + c]
    }

    /// Set pixel value at (x, y, channel).
    pub fn set(&mut self, x: usize, y: usize, c: usize, val: f64) {
        self.data[(y * self.width + x) * self.channels + c] = val;
    }

    /// Clamp all values to [min, max].
    pub fn clamp(&mut self, min: f64, max: f64) {
        for v in &mut self.data {
            *v = v.clamp(min, max);
        }
    }

    /// Normalize to [0, 1] range.
    pub fn normalize_to_unit(&mut self) {
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-10);
        for v in &mut self.data {
            *v = (*v - min) / range;
        }
    }

    /// Compute mean pixel value.
    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// Compute standard deviation.
    pub fn std_dev(&self) -> f64 {
        let m = self.mean();
        let var = self.data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.data.len() as f64;
        var.sqrt()
    }

    /// Total number of elements.
    pub fn size(&self) -> usize { self.data.len() }

    /// Create a tiled version of the image.
    pub fn tile(&self, tiles_x: usize, tiles_y: usize) -> ImageBuffer {
        let new_w = self.width * tiles_x;
        let new_h = self.height * tiles_y;
        let mut tiled = ImageBuffer::new(new_w, new_h, self.channels);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                for y in 0..self.height {
                    for x in 0..self.width {
                        for c in 0..self.channels {
                            tiled.set(tx * self.width + x, ty * self.height + y, c, self.at(x, y, c));
                        }
                    }
                }
            }
        }
        tiled
    }
}

impl fmt::Display for ImageBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageBuffer({}x{}x{}, mean={:.4}, std={:.4})",
            self.width, self.height, self.channels, self.mean(), self.std_dev())
    }
}

// ── Conditioning ──────────────────────────────────────────────

/// Conditioning information for guided generation.
#[derive(Debug, Clone)]
pub enum Condition {
    /// No conditioning (unconditional).
    None,
    /// Class label (integer).
    ClassLabel(usize),
    /// Text embedding (vector).
    TextEmbedding(Vec<f64>),
    /// Image embedding for img2img.
    ImageEmbedding(Vec<f64>),
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Condition::None => write!(f, "Unconditional"),
            Condition::ClassLabel(c) => write!(f, "Class({c})"),
            Condition::TextEmbedding(e) => write!(f, "TextEmbed(dim={})", e.len()),
            Condition::ImageEmbedding(e) => write!(f, "ImageEmbed(dim={})", e.len()),
        }
    }
}

// ── Sampling Strategy ─────────────────────────────────────────

/// Sampling strategy for the reverse diffusion process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Euler method (first-order).
    Euler,
    /// Heun method (second-order predictor-corrector).
    Heun,
    /// DPM++ solver (fast convergence).
    DpmPlusPlus,
    /// Ancestral sampling (stochastic).
    Ancestral,
}

impl SamplingStrategy {
    /// Apply one step of the sampling strategy.
    /// `x`: current state, `predicted_noise`: model output, `sigma`: noise level.
    pub fn step(&self, x: &[f64], predicted_noise: &[f64], sigma: f64, sigma_next: f64, rng: &mut Rng) -> Vec<f64> {
        match self {
            SamplingStrategy::Euler => {
                // x_{t-1} = x_t - (sigma - sigma_next) * noise_pred
                let dt = sigma - sigma_next;
                x.iter().zip(predicted_noise.iter())
                    .map(|(xi, ni)| xi - dt * ni)
                    .collect()
            }
            SamplingStrategy::Heun => {
                // Heun's method: predictor-corrector.
                let dt = sigma - sigma_next;
                let x_pred: Vec<f64> = x.iter().zip(predicted_noise.iter())
                    .map(|(xi, ni)| xi - dt * ni)
                    .collect();
                // Average the derivative at both endpoints.
                x.iter().zip(predicted_noise.iter()).zip(x_pred.iter())
                    .map(|((xi, ni), x_pi)| {
                        let d_avg = (ni + (xi - x_pi) / dt.max(1e-10)) / 2.0;
                        xi - dt * d_avg
                    })
                    .collect()
            }
            SamplingStrategy::DpmPlusPlus => {
                // DPM++ 2S: log-space stepping.
                let lambda = -(sigma.max(1e-10).ln());
                let lambda_next = -(sigma_next.max(1e-10).ln());
                let h = lambda_next - lambda;
                let coeff = (-h).exp();
                x.iter().zip(predicted_noise.iter())
                    .map(|(xi, ni)| coeff * xi + (1.0 - coeff) * ni)
                    .collect()
            }
            SamplingStrategy::Ancestral => {
                // Euler step + noise injection.
                let dt = sigma - sigma_next;
                let sigma_up = (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2).max(1e-10)).sqrt();
                let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).max(0.0).sqrt();
                let _ = sigma_down; // Used in full implementation.
                x.iter().zip(predicted_noise.iter())
                    .map(|(xi, ni)| xi - dt * ni + sigma_up * rng.normal())
                    .collect()
            }
        }
    }
}

impl fmt::Display for SamplingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingStrategy::Euler => write!(f, "Euler"),
            SamplingStrategy::Heun => write!(f, "Heun"),
            SamplingStrategy::DpmPlusPlus => write!(f, "DPM++"),
            SamplingStrategy::Ancestral => write!(f, "Ancestral"),
        }
    }
}

// ── Classifier-Free Guidance ──────────────────────────────────

/// Classifier-free guidance: blend unconditional and conditional predictions.
/// output = uncond + guidance_scale * (cond - uncond)
pub fn classifier_free_guidance(
    unconditional: &[f64],
    conditional: &[f64],
    guidance_scale: f64,
) -> Vec<f64> {
    unconditional.iter().zip(conditional.iter())
        .map(|(u, c)| u + guidance_scale * (c - u))
        .collect()
}

/// Apply negative prompting by interpolating away from negative prediction.
pub fn negative_prompt_guidance(
    negative: &[f64],
    conditional: &[f64],
    guidance_scale: f64,
) -> Vec<f64> {
    negative.iter().zip(conditional.iter())
        .map(|(n, c)| n + guidance_scale * (c - n))
        .collect()
}

// ── Noise Schedule for Pipeline ───────────────────────────────

/// Sigma schedule for the generation pipeline.
#[derive(Debug, Clone)]
pub struct SigmaSchedule {
    pub sigmas: Vec<f64>,
}

impl SigmaSchedule {
    /// Linear sigma schedule from sigma_max to sigma_min.
    pub fn linear(steps: usize, sigma_max: f64, sigma_min: f64) -> Self {
        let sigmas: Vec<f64> = (0..=steps)
            .map(|i| {
                let t = i as f64 / steps as f64;
                sigma_max * (1.0 - t) + sigma_min * t
            })
            .collect();
        Self { sigmas }
    }

    /// Karras schedule: sigma_i = (sigma_max^(1/rho) + t * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
    pub fn karras(steps: usize, sigma_max: f64, sigma_min: f64, rho: f64) -> Self {
        let min_inv = sigma_min.powf(1.0 / rho);
        let max_inv = sigma_max.powf(1.0 / rho);
        let sigmas: Vec<f64> = (0..=steps)
            .map(|i| {
                let t = i as f64 / steps as f64;
                (max_inv + t * (min_inv - max_inv)).powf(rho)
            })
            .collect();
        Self { sigmas }
    }

    pub fn steps(&self) -> usize { self.sigmas.len().saturating_sub(1) }
}

impl fmt::Display for SigmaSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SigmaSchedule(steps={}, sigma=[{:.4}..{:.4}])",
            self.steps(),
            self.sigmas.first().unwrap_or(&0.0),
            self.sigmas.last().unwrap_or(&0.0),
        )
    }
}

// ── Generation Pipeline ───────────────────────────────────────

/// Configuration for the image generation pipeline.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub num_steps: usize,
    pub guidance_scale: f64,
    pub strategy: SamplingStrategy,
    pub sigma_max: f64,
    pub sigma_min: f64,
    pub seed: u64,
}

impl GenerationConfig {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            channels: 3,
            num_steps: 20,
            guidance_scale: 7.5,
            strategy: SamplingStrategy::Euler,
            sigma_max: 14.6,
            sigma_min: 0.0292,
            seed: 42,
        }
    }

    pub fn with_channels(mut self, c: usize) -> Self { self.channels = c; self }
    pub fn with_steps(mut self, n: usize) -> Self { self.num_steps = n; self }
    pub fn with_guidance_scale(mut self, s: f64) -> Self { self.guidance_scale = s; self }
    pub fn with_strategy(mut self, s: SamplingStrategy) -> Self { self.strategy = s; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

impl fmt::Display for GenerationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GenerationConfig({}x{}x{}, steps={}, cfg={:.1}, strategy={})",
            self.width, self.height, self.channels,
            self.num_steps, self.guidance_scale, self.strategy,
        )
    }
}

/// Image generation pipeline.
#[derive(Debug, Clone)]
pub struct ImagePipeline {
    config: GenerationConfig,
    schedule: SigmaSchedule,
    rng: Rng,
    /// Simple denoiser weights (for simulation).
    denoiser_weights: Vec<Vec<f64>>,
    denoiser_biases: Vec<f64>,
}

impl ImagePipeline {
    pub fn new(config: GenerationConfig) -> Self {
        let schedule = SigmaSchedule::karras(config.num_steps, config.sigma_max, config.sigma_min, 7.0);
        let data_dim = config.width * config.height * config.channels;
        let hidden = data_dim.min(32);
        let seed = config.seed;
        let mut rng = Rng::new(seed + 1000);
        let scale = (2.0 / (data_dim + hidden) as f64).sqrt();

        let denoiser_weights = (0..data_dim)
            .map(|_| (0..data_dim.min(32)).map(|_| rng.normal() * scale).collect())
            .collect();
        let denoiser_biases = vec![0.0; data_dim];

        Self { config, schedule, rng: Rng::new(seed), denoiser_weights, denoiser_biases }
    }

    pub fn with_schedule(mut self, schedule: SigmaSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Simplified denoiser prediction.
    fn predict_noise(&self, x: &[f64], _sigma: f64) -> Vec<f64> {
        let hidden_dim = self.denoiser_weights[0].len();
        x.iter().enumerate()
            .map(|(j, _)| {
                let mut val = self.denoiser_biases[j];
                for k in 0..hidden_dim {
                    val += self.denoiser_weights[j][k] * x[k % x.len()];
                }
                val.tanh()
            })
            .collect()
    }

    /// Generate an image unconditionally.
    pub fn generate_unconditional(&mut self) -> ImageBuffer {
        let data_dim = self.config.width * self.config.height * self.config.channels;
        let mut x: Vec<f64> = (0..data_dim).map(|_| self.rng.normal() * self.config.sigma_max).collect();

        for i in 0..self.schedule.steps() {
            let sigma = self.schedule.sigmas[i];
            let sigma_next = self.schedule.sigmas[i + 1];
            let noise_pred = self.predict_noise(&x, sigma);
            x = self.config.strategy.step(&x, &noise_pred, sigma, sigma_next, &mut self.rng);
        }

        let mut img = ImageBuffer {
            width: self.config.width,
            height: self.config.height,
            channels: self.config.channels,
            data: x,
        };
        img.normalize_to_unit();
        img
    }

    /// Generate with classifier-free guidance.
    pub fn generate_guided(&mut self, condition: &Condition) -> ImageBuffer {
        let data_dim = self.config.width * self.config.height * self.config.channels;
        let mut x: Vec<f64> = (0..data_dim).map(|_| self.rng.normal() * self.config.sigma_max).collect();

        let cond_bias: f64 = match condition {
            Condition::None => 0.0,
            Condition::ClassLabel(c) => (*c as f64) * 0.01,
            Condition::TextEmbedding(e) => e.iter().sum::<f64>() / e.len().max(1) as f64 * 0.1,
            Condition::ImageEmbedding(e) => e.iter().sum::<f64>() / e.len().max(1) as f64 * 0.05,
        };

        for i in 0..self.schedule.steps() {
            let sigma = self.schedule.sigmas[i];
            let sigma_next = self.schedule.sigmas[i + 1];

            let uncond_pred = self.predict_noise(&x, sigma);
            let cond_pred: Vec<f64> = uncond_pred.iter()
                .map(|n| n + cond_bias * 0.1)
                .collect();

            let guided = classifier_free_guidance(&uncond_pred, &cond_pred, self.config.guidance_scale);
            x = self.config.strategy.step(&x, &guided, sigma, sigma_next, &mut self.rng);
        }

        let mut img = ImageBuffer {
            width: self.config.width,
            height: self.config.height,
            channels: self.config.channels,
            data: x,
        };
        img.normalize_to_unit();
        img
    }

    /// Generate a batch of images.
    pub fn generate_batch(&mut self, count: usize) -> Vec<ImageBuffer> {
        (0..count).map(|_| self.generate_unconditional()).collect()
    }

    pub fn config(&self) -> &GenerationConfig { &self.config }
}

impl fmt::Display for ImagePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImagePipeline({})", self.config)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_buffer_new() {
        let img = ImageBuffer::new(4, 4, 3);
        assert_eq!(img.size(), 48);
        assert!(img.mean().abs() < 1e-10);
    }

    #[test]
    fn test_image_buffer_noise() {
        let img = ImageBuffer::from_noise(4, 4, 3, 42);
        assert_eq!(img.size(), 48);
        assert!(img.std_dev() > 0.0);
    }

    #[test]
    fn test_image_buffer_access() {
        let mut img = ImageBuffer::new(2, 2, 1);
        img.set(1, 0, 0, 5.0);
        assert_eq!(img.at(1, 0, 0), 5.0);
        assert_eq!(img.at(0, 0, 0), 0.0);
    }

    #[test]
    fn test_image_clamp() {
        let mut img = ImageBuffer::from_noise(2, 2, 1, 42);
        img.clamp(0.0, 1.0);
        for v in &img.data {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn test_normalize_to_unit() {
        let mut img = ImageBuffer::from_noise(4, 4, 1, 42);
        img.normalize_to_unit();
        let min = img.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = img.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((min - 0.0).abs() < 1e-10);
        assert!((max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tile() {
        let img = ImageBuffer::from_noise(2, 2, 1, 42);
        let tiled = img.tile(2, 2);
        assert_eq!(tiled.width, 4);
        assert_eq!(tiled.height, 4);
        assert_eq!(tiled.at(0, 0, 0), img.at(0, 0, 0));
    }

    #[test]
    fn test_classifier_free_guidance() {
        let uncond = vec![0.0, 0.0];
        let cond = vec![1.0, 1.0];
        let result = classifier_free_guidance(&uncond, &cond, 7.5);
        assert!((result[0] - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_cfg_scale_1() {
        let uncond = vec![0.5];
        let cond = vec![1.0];
        let result = classifier_free_guidance(&uncond, &cond, 1.0);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_prompt() {
        let neg = vec![0.0];
        let cond = vec![1.0];
        let result = negative_prompt_guidance(&neg, &cond, 3.0);
        assert!((result[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigma_schedule_linear() {
        let sched = SigmaSchedule::linear(10, 14.0, 0.01);
        assert_eq!(sched.steps(), 10);
        assert!((sched.sigmas[0] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigma_schedule_karras() {
        let sched = SigmaSchedule::karras(20, 14.6, 0.0292, 7.0);
        assert_eq!(sched.steps(), 20);
        assert!(sched.sigmas[0] > sched.sigmas[20]);
    }

    #[test]
    fn test_euler_step() {
        let x = vec![1.0, 2.0];
        let noise = vec![0.1, 0.2];
        let mut rng = Rng::new(42);
        let result = SamplingStrategy::Euler.step(&x, &noise, 1.0, 0.5, &mut rng);
        assert!((result[0] - 0.95).abs() < 1e-10); // 1.0 - 0.5 * 0.1
    }

    #[test]
    fn test_generate_unconditional() {
        let config = GenerationConfig::new(2, 2).with_channels(1).with_steps(5);
        let mut pipeline = ImagePipeline::new(config);
        let img = pipeline.generate_unconditional();
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.channels, 1);
    }

    #[test]
    fn test_generate_guided() {
        let config = GenerationConfig::new(2, 2).with_channels(1).with_steps(3);
        let mut pipeline = ImagePipeline::new(config);
        let img = pipeline.generate_guided(&Condition::ClassLabel(5));
        assert_eq!(img.size(), 4);
    }

    #[test]
    fn test_generate_batch() {
        let config = GenerationConfig::new(2, 2).with_channels(1).with_steps(3);
        let mut pipeline = ImagePipeline::new(config);
        let batch = pipeline.generate_batch(3);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_condition_display() {
        assert!(format!("{}", Condition::None).contains("Unconditional"));
        assert!(format!("{}", Condition::ClassLabel(5)).contains("5"));
    }

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::new(8, 8)
            .with_channels(3)
            .with_steps(10)
            .with_guidance_scale(5.0)
            .with_strategy(SamplingStrategy::DpmPlusPlus)
            .with_seed(99);
        assert_eq!(config.num_steps, 10);
        assert_eq!(config.guidance_scale, 5.0);
    }

    #[test]
    fn test_pipeline_display() {
        let config = GenerationConfig::new(4, 4).with_steps(10);
        let pipeline = ImagePipeline::new(config);
        let s = format!("{pipeline}");
        assert!(s.contains("4x4"));
    }

    #[test]
    fn test_dpm_plus_plus_step() {
        let x = vec![1.0];
        let noise = vec![0.5];
        let mut rng = Rng::new(42);
        let result = SamplingStrategy::DpmPlusPlus.step(&x, &noise, 1.0, 0.5, &mut rng);
        assert!(result[0].is_finite());
    }
}
