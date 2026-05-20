//! Diffusion model: forward noising process, reverse denoising, noise schedule
//! (linear/cosine), DDPM/DDIM.
//!
//! Implements denoising diffusion probabilistic models with configurable noise
//! schedules, the forward diffusion process (q), reverse denoising process (p),
//! and both DDPM (stochastic) and DDIM (deterministic) sampling. Includes
//! alpha/beta schedule computation, signal-to-noise ratio tracking, and
//! simplified denoiser networks for the reverse process.

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

// ── Noise Schedule ────────────────────────────────────────────

/// Type of noise schedule for the diffusion process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseSchedule {
    /// Linear schedule from beta_start to beta_end.
    Linear,
    /// Cosine schedule (improved denoising for low-step regions).
    Cosine,
    /// Quadratic schedule (beta grows quadratically).
    Quadratic,
}

impl fmt::Display for NoiseSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoiseSchedule::Linear => write!(f, "Linear"),
            NoiseSchedule::Cosine => write!(f, "Cosine"),
            NoiseSchedule::Quadratic => write!(f, "Quadratic"),
        }
    }
}

/// Precomputed schedule values for all timesteps.
#[derive(Debug, Clone)]
pub struct DiffusionSchedule {
    /// Number of diffusion timesteps.
    pub num_steps: usize,
    /// Beta values (noise variance per step).
    pub betas: Vec<f64>,
    /// Alpha = 1 - beta.
    pub alphas: Vec<f64>,
    /// Cumulative product of alphas: alpha_bar_t = prod(alpha_1..alpha_t).
    pub alpha_bars: Vec<f64>,
    /// sqrt(alpha_bar_t).
    pub sqrt_alpha_bars: Vec<f64>,
    /// sqrt(1 - alpha_bar_t).
    pub sqrt_one_minus_alpha_bars: Vec<f64>,
    /// Signal-to-noise ratio at each step.
    pub snr: Vec<f64>,
}

impl DiffusionSchedule {
    /// Create a schedule with the given parameters.
    pub fn new(schedule_type: NoiseSchedule, num_steps: usize, beta_start: f64, beta_end: f64) -> Self {
        let betas: Vec<f64> = match schedule_type {
            NoiseSchedule::Linear => {
                (0..num_steps)
                    .map(|i| beta_start + (beta_end - beta_start) * i as f64 / (num_steps - 1).max(1) as f64)
                    .collect()
            }
            NoiseSchedule::Cosine => {
                let s = 0.008;
                let max_beta = 0.999;
                (0..num_steps)
                    .map(|i| {
                        let t1 = i as f64 / num_steps as f64;
                        let t2 = (i + 1) as f64 / num_steps as f64;
                        let f1 = ((t1 + s) / (1.0 + s) * std::f64::consts::FRAC_PI_2).cos().powi(2);
                        let f2 = ((t2 + s) / (1.0 + s) * std::f64::consts::FRAC_PI_2).cos().powi(2);
                        (1.0 - f2 / f1).clamp(0.0, max_beta)
                    })
                    .collect()
            }
            NoiseSchedule::Quadratic => {
                let sqrt_start = beta_start.sqrt();
                let sqrt_end = beta_end.sqrt();
                (0..num_steps)
                    .map(|i| {
                        let t = i as f64 / (num_steps - 1).max(1) as f64;
                        let v = sqrt_start + (sqrt_end - sqrt_start) * t;
                        v * v
                    })
                    .collect()
            }
        };

        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

        let mut alpha_bars = Vec::with_capacity(num_steps);
        let mut cum_prod = 1.0;
        for &a in &alphas {
            cum_prod *= a;
            alpha_bars.push(cum_prod);
        }

        let sqrt_alpha_bars: Vec<f64> = alpha_bars.iter().map(|ab| ab.sqrt()).collect();
        let sqrt_one_minus_alpha_bars: Vec<f64> = alpha_bars.iter().map(|ab| (1.0 - ab).sqrt()).collect();
        let snr: Vec<f64> = alpha_bars.iter().map(|ab| {
            let noise_var = (1.0 - ab).max(1e-15);
            ab / noise_var
        }).collect();

        Self { num_steps, betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, snr }
    }

    /// Get the signal-to-noise ratio at timestep t (in dB).
    pub fn snr_db(&self, t: usize) -> f64 {
        10.0 * self.snr[t.min(self.num_steps - 1)].max(1e-15).log10()
    }
}

impl fmt::Display for DiffusionSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DiffusionSchedule(steps={}, beta=[{:.6}..{:.6}], alpha_bar=[{:.4}..{:.4}])",
            self.num_steps,
            self.betas[0],
            self.betas[self.num_steps - 1],
            self.alpha_bars[0],
            self.alpha_bars[self.num_steps - 1],
        )
    }
}

// ── Forward Process ───────────────────────────────────────────

/// Forward diffusion: adds noise to data according to the schedule.
pub fn forward_diffusion(
    x0: &[f64],
    t: usize,
    schedule: &DiffusionSchedule,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>) {
    let t = t.min(schedule.num_steps - 1);
    let sqrt_ab = schedule.sqrt_alpha_bars[t];
    let sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bars[t];

    let noise: Vec<f64> = (0..x0.len()).map(|_| rng.normal()).collect();
    let noisy: Vec<f64> = x0.iter().zip(noise.iter())
        .map(|(x, n)| sqrt_ab * x + sqrt_one_minus_ab * n)
        .collect();

    (noisy, noise)
}

/// Compute the noise level (sigma) at timestep t.
pub fn noise_level(t: usize, schedule: &DiffusionSchedule) -> f64 {
    let t = t.min(schedule.num_steps - 1);
    schedule.sqrt_one_minus_alpha_bars[t]
}

// ── Denoiser Network ──────────────────────────────────────────

/// Simple MLP denoiser that predicts noise from noisy input + timestep.
#[derive(Debug, Clone)]
pub struct Denoiser {
    weights_hidden: Vec<Vec<f64>>,
    biases_hidden: Vec<f64>,
    weights_out: Vec<Vec<f64>>,
    biases_out: Vec<f64>,
    data_dim: usize,
    hidden_dim: usize,
}

impl Denoiser {
    pub fn new(data_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let input_dim = data_dim + 1; // data + timestep embedding.
        let scale_h = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale_o = (2.0 / (hidden_dim + data_dim) as f64).sqrt();

        let weights_hidden: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| (0..input_dim).map(|_| rng.normal() * scale_h).collect())
            .collect();
        let biases_hidden = vec![0.0; hidden_dim];

        let weights_out: Vec<Vec<f64>> = (0..data_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.normal() * scale_o).collect())
            .collect();
        let biases_out = vec![0.0; data_dim];

        Self { weights_hidden, biases_hidden, weights_out, biases_out, data_dim, hidden_dim }
    }

    /// Predict noise given noisy input and normalized timestep.
    pub fn predict_noise(&self, x_noisy: &[f64], t_normalized: f64) -> Vec<f64> {
        // Concatenate input with timestep.
        let mut input = x_noisy.to_vec();
        input.push(t_normalized);

        // Hidden layer with ReLU.
        let hidden: Vec<f64> = (0..self.hidden_dim)
            .map(|j| {
                let z: f64 = self.weights_hidden[j].iter().zip(input.iter())
                    .map(|(w, x)| w * x).sum::<f64>() + self.biases_hidden[j];
                z.max(0.0)
            })
            .collect();

        // Output layer (linear).
        (0..self.data_dim)
            .map(|j| {
                self.weights_out[j].iter().zip(hidden.iter())
                    .map(|(w, h)| w * h).sum::<f64>() + self.biases_out[j]
            })
            .collect()
    }

    pub fn param_count(&self) -> usize {
        let h = self.weights_hidden.len() * self.weights_hidden[0].len() + self.biases_hidden.len();
        let o = self.weights_out.len() * self.weights_out[0].len() + self.biases_out.len();
        h + o
    }
}

impl fmt::Display for Denoiser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Denoiser(data_dim={}, hidden={}, params={})",
            self.data_dim, self.hidden_dim, self.param_count())
    }
}

// ── DDPM Sampler ──────────────────────────────────────────────

/// DDPM (Denoising Diffusion Probabilistic Model) sampler.
#[derive(Debug, Clone)]
pub struct DdpmSampler {
    schedule: DiffusionSchedule,
    denoiser: Denoiser,
    rng: Rng,
}

impl DdpmSampler {
    pub fn new(schedule: DiffusionSchedule, denoiser: Denoiser, seed: u64) -> Self {
        Self { schedule, denoiser, rng: Rng::new(seed) }
    }

    /// Sample from pure noise using DDPM reverse process.
    pub fn sample(&mut self, data_dim: usize) -> Vec<f64> {
        let mut x: Vec<f64> = (0..data_dim).map(|_| self.rng.normal()).collect();

        for t in (0..self.schedule.num_steps).rev() {
            let t_norm = t as f64 / self.schedule.num_steps as f64;
            let predicted_noise = self.denoiser.predict_noise(&x, t_norm);

            let alpha = self.schedule.alphas[t];
            let alpha_bar = self.schedule.alpha_bars[t];
            let beta = self.schedule.betas[t];

            let coeff1 = 1.0 / alpha.sqrt();
            let coeff2 = beta / (1.0 - alpha_bar).max(1e-10).sqrt();

            let noise: Vec<f64> = if t > 0 {
                let sigma = beta.sqrt();
                (0..data_dim).map(|_| sigma * self.rng.normal()).collect()
            } else {
                vec![0.0; data_dim]
            };

            x = x.iter().zip(predicted_noise.iter()).zip(noise.iter())
                .map(|((xi, ni), zi)| coeff1 * (xi - coeff2 * ni) + zi)
                .collect();
        }

        x
    }

    /// Sample multiple images.
    pub fn sample_batch(&mut self, data_dim: usize, count: usize) -> Vec<Vec<f64>> {
        (0..count).map(|_| self.sample(data_dim)).collect()
    }
}

impl fmt::Display for DdpmSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DdpmSampler(steps={}, {})", self.schedule.num_steps, self.denoiser)
    }
}

// ── DDIM Sampler ──────────────────────────────────────────────

/// DDIM (Denoising Diffusion Implicit Model) sampler for faster deterministic sampling.
#[derive(Debug, Clone)]
pub struct DdimSampler {
    schedule: DiffusionSchedule,
    denoiser: Denoiser,
    /// Number of actual sampling steps (can be much less than training steps).
    sampling_steps: usize,
    /// Eta parameter: 0 = fully deterministic, 1 = equivalent to DDPM.
    eta: f64,
    rng: Rng,
}

impl DdimSampler {
    pub fn new(schedule: DiffusionSchedule, denoiser: Denoiser, sampling_steps: usize, seed: u64) -> Self {
        Self { schedule, denoiser, sampling_steps, eta: 0.0, rng: Rng::new(seed) }
    }

    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta.clamp(0.0, 1.0);
        self
    }

    /// Build the subsequence of timesteps for accelerated sampling.
    fn timestep_subsequence(&self) -> Vec<usize> {
        let total = self.schedule.num_steps;
        let steps = self.sampling_steps.min(total);
        if steps == 0 { return vec![]; }
        (0..steps)
            .map(|i| ((total - 1) as f64 * i as f64 / (steps - 1).max(1) as f64) as usize)
            .collect()
    }

    /// Sample using the DDIM update rule.
    pub fn sample(&mut self, data_dim: usize) -> Vec<f64> {
        let mut x: Vec<f64> = (0..data_dim).map(|_| self.rng.normal()).collect();
        let timesteps = self.timestep_subsequence();

        for (idx, &t) in timesteps.iter().rev().enumerate() {
            let t_norm = t as f64 / self.schedule.num_steps as f64;
            let eps = self.denoiser.predict_noise(&x, t_norm);

            let alpha_bar_t = self.schedule.alpha_bars[t];
            let alpha_bar_prev = if idx < timesteps.len() - 1 {
                let prev_t = timesteps[timesteps.len() - 2 - idx];
                self.schedule.alpha_bars[prev_t]
            } else {
                1.0 // alpha_bar_0 = 1.
            };

            // Predicted x0.
            let x0_pred: Vec<f64> = x.iter().zip(eps.iter())
                .map(|(xi, ei)| (xi - (1.0 - alpha_bar_t).max(1e-10).sqrt() * ei) / alpha_bar_t.max(1e-10).sqrt())
                .collect();

            // DDIM update.
            let sigma = self.eta * ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).max(1e-10) * (1.0 - alpha_bar_t / alpha_bar_prev)).max(0.0).sqrt();

            let dir_coeff = (1.0 - alpha_bar_prev - sigma * sigma).max(0.0).sqrt();

            x = x0_pred.iter().zip(eps.iter())
                .map(|(x0, ei)| {
                    let noise = if sigma > 1e-10 { self.rng.normal() } else { 0.0 };
                    alpha_bar_prev.sqrt() * x0 + dir_coeff * ei + sigma * noise
                })
                .collect();
        }

        x
    }

    pub fn sample_batch(&mut self, data_dim: usize, count: usize) -> Vec<Vec<f64>> {
        (0..count).map(|_| self.sample(data_dim)).collect()
    }
}

impl fmt::Display for DdimSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DdimSampler(sampling_steps={}, eta={:.2}, total_steps={})",
            self.sampling_steps, self.eta, self.schedule.num_steps)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_schedule_monotonic() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 100, 1e-4, 0.02);
        for i in 1..sched.betas.len() {
            assert!(sched.betas[i] >= sched.betas[i - 1]);
        }
    }

    #[test]
    fn test_cosine_schedule_bounded() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Cosine, 100, 1e-4, 0.02);
        for &b in &sched.betas {
            assert!(b >= 0.0 && b <= 1.0);
        }
    }

    #[test]
    fn test_alpha_bars_decreasing() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 50, 1e-4, 0.02);
        for i in 1..sched.alpha_bars.len() {
            assert!(sched.alpha_bars[i] <= sched.alpha_bars[i - 1]);
        }
    }

    #[test]
    fn test_sqrt_identity() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 20, 1e-4, 0.02);
        for i in 0..20 {
            let sum = sched.sqrt_alpha_bars[i].powi(2) + sched.sqrt_one_minus_alpha_bars[i].powi(2);
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_snr_decreasing() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 50, 1e-4, 0.02);
        for i in 1..sched.snr.len() {
            assert!(sched.snr[i] <= sched.snr[i - 1]);
        }
    }

    #[test]
    fn test_snr_db() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 10, 1e-4, 0.02);
        let db = sched.snr_db(0);
        assert!(db > 0.0); // Early steps have high SNR.
    }

    #[test]
    fn test_forward_diffusion_shape() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 100, 1e-4, 0.02);
        let x0 = vec![1.0, 2.0, 3.0];
        let mut rng = Rng::new(42);
        let (noisy, noise) = forward_diffusion(&x0, 50, &sched, &mut rng);
        assert_eq!(noisy.len(), 3);
        assert_eq!(noise.len(), 3);
    }

    #[test]
    fn test_forward_diffusion_t0_close() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 100, 1e-4, 0.02);
        let x0 = vec![1.0, 0.0, -1.0];
        let mut rng = Rng::new(42);
        let (noisy, _) = forward_diffusion(&x0, 0, &sched, &mut rng);
        // At t=0, very little noise is added.
        for (xi, ni) in x0.iter().zip(noisy.iter()) {
            assert!((xi - ni).abs() < 0.5);
        }
    }

    #[test]
    fn test_noise_level() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 100, 1e-4, 0.02);
        let nl_early = noise_level(0, &sched);
        let nl_late = noise_level(99, &sched);
        assert!(nl_late > nl_early);
    }

    #[test]
    fn test_denoiser_output_shape() {
        let denoiser = Denoiser::new(4, 8, 42);
        let noisy = vec![0.5, -0.3, 0.1, 0.8];
        let pred = denoiser.predict_noise(&noisy, 0.5);
        assert_eq!(pred.len(), 4);
    }

    #[test]
    fn test_denoiser_param_count() {
        let denoiser = Denoiser::new(3, 5, 42);
        // Hidden: (3+1)*5 + 5 = 25, Out: 5*3 + 3 = 18. Total = 43.
        assert_eq!(denoiser.param_count(), 43);
    }

    #[test]
    fn test_ddpm_sample_shape() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 10, 1e-4, 0.02);
        let denoiser = Denoiser::new(3, 8, 42);
        let mut sampler = DdpmSampler::new(sched, denoiser, 123);
        let sample = sampler.sample(3);
        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn test_ddpm_batch() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 5, 1e-4, 0.02);
        let denoiser = Denoiser::new(2, 4, 42);
        let mut sampler = DdpmSampler::new(sched, denoiser, 123);
        let batch = sampler.sample_batch(2, 3);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_ddim_sample_shape() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 20, 1e-4, 0.02);
        let denoiser = Denoiser::new(3, 6, 42);
        let mut sampler = DdimSampler::new(sched, denoiser, 5, 99);
        let sample = sampler.sample(3);
        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn test_ddim_deterministic() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 10, 1e-4, 0.02);
        let denoiser = Denoiser::new(2, 4, 42);
        let mut s1 = DdimSampler::new(sched.clone(), denoiser.clone(), 3, 99).with_eta(0.0);
        let mut s2 = DdimSampler::new(sched, denoiser, 3, 99).with_eta(0.0);
        let a = s1.sample(2);
        let b = s2.sample(2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_quadratic_schedule() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Quadratic, 50, 1e-4, 0.02);
        assert_eq!(sched.betas.len(), 50);
        assert!(sched.betas[49] > sched.betas[0]);
    }

    #[test]
    fn test_schedule_display() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 100, 1e-4, 0.02);
        let s = format!("{sched}");
        assert!(s.contains("steps=100"));
    }

    #[test]
    fn test_ddpm_display() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 10, 1e-4, 0.02);
        let denoiser = Denoiser::new(3, 8, 42);
        let sampler = DdpmSampler::new(sched, denoiser, 123);
        let s = format!("{sampler}");
        assert!(s.contains("DdpmSampler"));
    }

    #[test]
    fn test_ddim_display() {
        let sched = DiffusionSchedule::new(NoiseSchedule::Linear, 100, 1e-4, 0.02);
        let denoiser = Denoiser::new(3, 8, 42);
        let sampler = DdimSampler::new(sched, denoiser, 10, 99);
        let s = format!("{sampler}");
        assert!(s.contains("sampling_steps=10"));
    }
}
