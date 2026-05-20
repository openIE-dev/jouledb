//! Gradient descent optimizers for neural network training.
//!
//! Provides first-order optimization algorithms that update model parameters
//! by following the negative gradient of the loss function:
//!
//! - [`SgdOptimizer`] — vanilla SGD and mini-batch SGD
//! - [`MomentumOptimizer`] — SGD with classical momentum
//! - [`NesterovOptimizer`] — Nesterov accelerated gradient
//! - [`GradientClipper`] — gradient clipping by value or norm
//! - [`MiniBatchSampler`] — index-based mini-batch generation

use std::fmt;

// ── Gradient Clipping ──────────────────────────────────────────────

/// Strategy for constraining gradient magnitudes before parameter updates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipStrategy {
    /// Clamp each element to `[-threshold, threshold]`.
    ByValue(f64),
    /// Scale the entire gradient vector so its L2 norm ≤ threshold.
    ByNorm(f64),
    /// Scale by global norm across all parameter groups.
    ByGlobalNorm(f64),
    /// No clipping applied.
    None,
}

/// Applies gradient clipping in-place according to a chosen [`ClipStrategy`].
pub struct GradientClipper {
    strategy: ClipStrategy,
    total_clips: u64,
}

impl GradientClipper {
    pub fn new(strategy: ClipStrategy) -> Self {
        Self {
            strategy,
            total_clips: 0,
        }
    }

    /// Clip `grads` in-place. Returns `true` if any modification was made.
    pub fn clip(&mut self, grads: &mut [f64]) -> bool {
        match self.strategy {
            ClipStrategy::ByValue(threshold) => {
                let mut clipped = false;
                for g in grads.iter_mut() {
                    if *g > threshold {
                        *g = threshold;
                        clipped = true;
                    } else if *g < -threshold {
                        *g = -threshold;
                        clipped = true;
                    }
                }
                if clipped {
                    self.total_clips += 1;
                }
                clipped
            }
            ClipStrategy::ByNorm(max_norm) => {
                let norm = vector_l2_norm(grads);
                if norm > max_norm && norm > 0.0 {
                    let scale = max_norm / norm;
                    for g in grads.iter_mut() {
                        *g *= scale;
                    }
                    self.total_clips += 1;
                    true
                } else {
                    false
                }
            }
            ClipStrategy::ByGlobalNorm(max_norm) => {
                // For a single vector, global norm == L2 norm.
                let norm = vector_l2_norm(grads);
                if norm > max_norm && norm > 0.0 {
                    let scale = max_norm / norm;
                    for g in grads.iter_mut() {
                        *g *= scale;
                    }
                    self.total_clips += 1;
                    true
                } else {
                    false
                }
            }
            ClipStrategy::None => false,
        }
    }

    /// Clip across multiple gradient groups sharing a single global norm budget.
    pub fn clip_global(&mut self, groups: &mut [&mut [f64]]) -> bool {
        if let ClipStrategy::ByGlobalNorm(max_norm) = self.strategy {
            let mut sum_sq = 0.0_f64;
            for group in groups.iter() {
                for g in group.iter() {
                    sum_sq += g * g;
                }
            }
            let global_norm = sum_sq.sqrt();
            if global_norm > max_norm && global_norm > 0.0 {
                let scale = max_norm / global_norm;
                for group in groups.iter_mut() {
                    for g in group.iter_mut() {
                        *g *= scale;
                    }
                }
                self.total_clips += 1;
                return true;
            }
            false
        } else {
            let mut any = false;
            for group in groups.iter_mut() {
                any |= self.clip(group);
            }
            any
        }
    }

    pub fn total_clips(&self) -> u64 {
        self.total_clips
    }

    pub fn strategy(&self) -> ClipStrategy {
        self.strategy
    }
}

impl fmt::Display for GradientClipper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.strategy {
            ClipStrategy::ByValue(t) => write!(f, "GradientClipper(value≤{t:.4})"),
            ClipStrategy::ByNorm(t) => write!(f, "GradientClipper(norm≤{t:.4})"),
            ClipStrategy::ByGlobalNorm(t) => write!(f, "GradientClipper(global_norm≤{t:.4})"),
            ClipStrategy::None => write!(f, "GradientClipper(none)"),
        }
    }
}

// ── SGD Optimizer ──────────────────────────────────────────────────

/// Vanilla Stochastic Gradient Descent.
///
/// Update rule: `θ ← θ − lr * ∇L(θ)`
pub struct SgdOptimizer {
    learning_rate: f64,
    clipper: GradientClipper,
    step_count: u64,
}

impl SgdOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            clipper: GradientClipper::new(ClipStrategy::None),
            step_count: 0,
        }
    }

    pub fn with_clip(mut self, strategy: ClipStrategy) -> Self {
        self.clipper = GradientClipper::new(strategy);
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Perform a single parameter update step.
    /// `params` and `grads` must have the same length.
    pub fn step(&mut self, params: &mut [f64], grads: &mut [f64]) {
        assert_eq!(params.len(), grads.len(), "params and grads length mismatch");
        self.clipper.clip(grads);
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            *p -= self.learning_rate * g;
        }
        self.step_count += 1;
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

impl fmt::Display for SgdOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SGD(lr={:.6}, steps={})", self.learning_rate, self.step_count)
    }
}

// ── Momentum Optimizer ─────────────────────────────────────────────

/// SGD with classical (Polyak) momentum.
///
/// Update rule:
/// ```text
/// v ← μ * v + ∇L(θ)
/// θ ← θ − lr * v
/// ```
pub struct MomentumOptimizer {
    learning_rate: f64,
    momentum: f64,
    velocity: Vec<f64>,
    clipper: GradientClipper,
    step_count: u64,
    dampening: f64,
}

impl MomentumOptimizer {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: Vec::new(),
            clipper: GradientClipper::new(ClipStrategy::None),
            step_count: 0,
            dampening: 0.0,
        }
    }

    pub fn with_clip(mut self, strategy: ClipStrategy) -> Self {
        self.clipper = GradientClipper::new(strategy);
        self
    }

    pub fn with_dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn step(&mut self, params: &mut [f64], grads: &mut [f64]) {
        assert_eq!(params.len(), grads.len());
        self.clipper.clip(grads);

        if self.velocity.is_empty() {
            self.velocity = vec![0.0; params.len()];
        }
        assert_eq!(self.velocity.len(), params.len());

        for i in 0..params.len() {
            self.velocity[i] = self.momentum * self.velocity[i]
                + (1.0 - self.dampening) * grads[i];
            params[i] -= self.learning_rate * self.velocity[i];
        }
        self.step_count += 1;
    }

    pub fn velocity(&self) -> &[f64] {
        &self.velocity
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn reset(&mut self) {
        self.velocity.clear();
        self.step_count = 0;
    }
}

impl fmt::Display for MomentumOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Momentum(lr={:.6}, μ={:.4}, steps={})",
            self.learning_rate, self.momentum, self.step_count
        )
    }
}

// ── Nesterov Accelerated Gradient ──────────────────────────────────

/// Nesterov momentum (look-ahead gradient).
///
/// Update rule:
/// ```text
/// v ← μ * v + ∇L(θ + μ * v)      [conceptually]
/// θ ← θ − lr * v
/// ```
///
/// In practice we use the reformulated version that avoids the look-ahead
/// evaluation and instead applies a correction after the gradient step:
/// ```text
/// v ← μ * v + g
/// θ ← θ − lr * (g + μ * v)
/// ```
pub struct NesterovOptimizer {
    learning_rate: f64,
    momentum: f64,
    velocity: Vec<f64>,
    clipper: GradientClipper,
    step_count: u64,
}

impl NesterovOptimizer {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: Vec::new(),
            clipper: GradientClipper::new(ClipStrategy::None),
            step_count: 0,
        }
    }

    pub fn with_clip(mut self, strategy: ClipStrategy) -> Self {
        self.clipper = GradientClipper::new(strategy);
        self
    }

    pub fn step(&mut self, params: &mut [f64], grads: &mut [f64]) {
        assert_eq!(params.len(), grads.len());
        self.clipper.clip(grads);

        if self.velocity.is_empty() {
            self.velocity = vec![0.0; params.len()];
        }

        for i in 0..params.len() {
            self.velocity[i] = self.momentum * self.velocity[i] + grads[i];
            // Nesterov correction: use gradient + momentum * updated velocity
            params[i] -= self.learning_rate * (grads[i] + self.momentum * self.velocity[i]);
        }
        self.step_count += 1;
    }

    pub fn velocity(&self) -> &[f64] {
        &self.velocity
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

impl fmt::Display for NesterovOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Nesterov(lr={:.6}, μ={:.4}, steps={})",
            self.learning_rate, self.momentum, self.step_count
        )
    }
}

// ── Mini-Batch Sampler ─────────────────────────────────────────────

/// Generates index batches for mini-batch SGD.
///
/// Given a dataset of `n` samples and a `batch_size`, produces sequential
/// index slices. Supports optional shuffling via a simple LCG PRNG.
pub struct MiniBatchSampler {
    dataset_size: usize,
    batch_size: usize,
    indices: Vec<usize>,
    cursor: usize,
    epoch: u64,
    rng_state: u64,
    shuffle: bool,
}

impl MiniBatchSampler {
    pub fn new(dataset_size: usize, batch_size: usize) -> Self {
        let indices: Vec<usize> = (0..dataset_size).collect();
        Self {
            dataset_size,
            batch_size: batch_size.min(dataset_size).max(1),
            indices,
            cursor: 0,
            epoch: 0,
            rng_state: 42,
            shuffle: false,
        }
    }

    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle = true;
        self.rng_state = seed;
        self.reshuffle();
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.min(self.dataset_size).max(1);
        self
    }

    /// Returns the next batch of indices, advancing the cursor.
    /// When the dataset is exhausted, wraps to the next epoch.
    pub fn next_batch(&mut self) -> &[usize] {
        if self.cursor >= self.dataset_size {
            self.epoch += 1;
            self.cursor = 0;
            if self.shuffle {
                self.reshuffle();
            }
        }
        let end = (self.cursor + self.batch_size).min(self.dataset_size);
        let batch = &self.indices[self.cursor..end];
        self.cursor = end;
        batch
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn batches_per_epoch(&self) -> usize {
        (self.dataset_size + self.batch_size - 1) / self.batch_size
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
        self.epoch = 0;
    }

    /// Fisher-Yates shuffle using a simple LCG.
    fn reshuffle(&mut self) {
        let n = self.indices.len();
        for i in (1..n).rev() {
            self.rng_state = self.rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (self.rng_state >> 33) as usize % (i + 1);
            self.indices.swap(i, j);
        }
    }
}

impl fmt::Display for MiniBatchSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MiniBatchSampler(n={}, batch={}, epoch={})",
            self.dataset_size, self.batch_size, self.epoch
        )
    }
}

// ── Gradient Accumulator ───────────────────────────────────────────

/// Accumulates gradients over multiple micro-batches before a single
/// optimizer step, effectively simulating a larger batch size.
pub struct GradientAccumulator {
    accum: Vec<f64>,
    accum_steps: usize,
    target_steps: usize,
    current_step: usize,
}

impl GradientAccumulator {
    pub fn new(param_count: usize, accumulation_steps: usize) -> Self {
        Self {
            accum: vec![0.0; param_count],
            accum_steps: 0,
            target_steps: accumulation_steps.max(1),
            current_step: 0,
        }
    }

    /// Add a micro-batch gradient. Returns `true` when accumulation is
    /// complete and the caller should perform an optimizer step.
    pub fn accumulate(&mut self, grads: &[f64]) -> bool {
        assert_eq!(grads.len(), self.accum.len());
        for (a, g) in self.accum.iter_mut().zip(grads.iter()) {
            *a += g;
        }
        self.accum_steps += 1;
        self.current_step += 1;
        self.accum_steps >= self.target_steps
    }

    /// Retrieve averaged accumulated gradients and reset the accumulator.
    pub fn take_averaged(&mut self) -> Vec<f64> {
        let scale = 1.0 / self.accum_steps.max(1) as f64;
        let result: Vec<f64> = self.accum.iter().map(|a| a * scale).collect();
        self.accum.iter_mut().for_each(|a| *a = 0.0);
        self.accum_steps = 0;
        result
    }

    pub fn is_ready(&self) -> bool {
        self.accum_steps >= self.target_steps
    }

    pub fn accumulated_steps(&self) -> usize {
        self.accum_steps
    }

    pub fn total_steps(&self) -> usize {
        self.current_step
    }
}

impl fmt::Display for GradientAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradientAccumulator({}/{} steps, total={})",
            self.accum_steps, self.target_steps, self.current_step
        )
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn vector_l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_basic_step() {
        let mut opt = SgdOptimizer::new(0.1);
        let mut params = vec![1.0, 2.0, 3.0];
        let mut grads = vec![0.5, 1.0, 1.5];
        opt.step(&mut params, &mut grads);
        assert!((params[0] - 0.95).abs() < 1e-10);
        assert!((params[1] - 1.9).abs() < 1e-10);
        assert!((params[2] - 2.85).abs() < 1e-10);
    }

    #[test]
    fn sgd_step_count_increments() {
        let mut opt = SgdOptimizer::new(0.01);
        let mut p = vec![0.0];
        let mut g = vec![1.0];
        opt.step(&mut p, &mut g);
        opt.step(&mut p, &mut g);
        assert_eq!(opt.step_count(), 2);
    }

    #[test]
    fn sgd_display() {
        let opt = SgdOptimizer::new(0.001);
        let s = format!("{opt}");
        assert!(s.contains("SGD"));
        assert!(s.contains("0.001"));
    }

    #[test]
    fn momentum_converges_faster() {
        let mut sgd = SgdOptimizer::new(0.1);
        let mut mom = MomentumOptimizer::new(0.1, 0.9);
        let mut p_sgd = vec![5.0];
        let mut p_mom = vec![5.0];
        // Repeated steps with constant gradient
        for _ in 0..10 {
            sgd.step(&mut p_sgd, &mut vec![1.0]);
            mom.step(&mut p_mom, &mut vec![1.0]);
        }
        // Momentum should have moved further from 5.0
        assert!(p_mom[0] < p_sgd[0]);
    }

    #[test]
    fn momentum_velocity_initialized() {
        let mut opt = MomentumOptimizer::new(0.1, 0.9);
        assert!(opt.velocity().is_empty());
        opt.step(&mut vec![1.0], &mut vec![0.5]);
        assert_eq!(opt.velocity().len(), 1);
    }

    #[test]
    fn momentum_reset() {
        let mut opt = MomentumOptimizer::new(0.1, 0.9);
        opt.step(&mut vec![1.0], &mut vec![0.5]);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
        assert!(opt.velocity().is_empty());
    }

    #[test]
    fn nesterov_differs_from_classical() {
        let mut mom = MomentumOptimizer::new(0.1, 0.9);
        let mut nes = NesterovOptimizer::new(0.1, 0.9);
        let mut p_m = vec![3.0];
        let mut p_n = vec![3.0];
        for _ in 0..5 {
            mom.step(&mut p_m, &mut vec![1.0]);
            nes.step(&mut p_n, &mut vec![1.0]);
        }
        // They should produce different trajectories
        assert!((p_m[0] - p_n[0]).abs() > 1e-6);
    }

    #[test]
    fn clip_by_value() {
        let mut clipper = GradientClipper::new(ClipStrategy::ByValue(1.0));
        let mut grads = vec![-5.0, 0.5, 3.0];
        let clipped = clipper.clip(&mut grads);
        assert!(clipped);
        assert_eq!(grads, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn clip_by_norm() {
        let mut clipper = GradientClipper::new(ClipStrategy::ByNorm(1.0));
        let mut grads = vec![3.0, 4.0]; // norm = 5
        clipper.clip(&mut grads);
        let norm = vector_l2_norm(&grads);
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn clip_by_norm_no_change() {
        let mut clipper = GradientClipper::new(ClipStrategy::ByNorm(10.0));
        let mut grads = vec![3.0, 4.0]; // norm = 5 < 10
        let clipped = clipper.clip(&mut grads);
        assert!(!clipped);
        assert!((grads[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn clip_none() {
        let mut clipper = GradientClipper::new(ClipStrategy::None);
        let mut grads = vec![100.0, -200.0];
        let clipped = clipper.clip(&mut grads);
        assert!(!clipped);
    }

    #[test]
    fn clip_global_norm_multi_group() {
        let mut clipper = GradientClipper::new(ClipStrategy::ByGlobalNorm(1.0));
        let mut g1 = vec![3.0, 0.0];
        let mut g2 = vec![0.0, 4.0];
        // Global norm = sqrt(9+16) = 5
        clipper.clip_global(&mut [&mut g1, &mut g2]);
        let global = (g1.iter().chain(g2.iter()).map(|x| x * x).sum::<f64>()).sqrt();
        assert!((global - 1.0).abs() < 1e-10);
    }

    #[test]
    fn clipper_display() {
        let c = GradientClipper::new(ClipStrategy::ByValue(2.5));
        assert!(format!("{c}").contains("value"));
    }

    #[test]
    fn mini_batch_basic() {
        let mut sampler = MiniBatchSampler::new(10, 3);
        let b1 = sampler.next_batch().to_vec();
        assert_eq!(b1.len(), 3);
        let b2 = sampler.next_batch().to_vec();
        assert_eq!(b2.len(), 3);
        let b3 = sampler.next_batch().to_vec();
        assert_eq!(b3.len(), 3);
        let b4 = sampler.next_batch().to_vec();
        assert_eq!(b4.len(), 1); // remainder
    }

    #[test]
    fn mini_batch_epoch_rolls() {
        let mut sampler = MiniBatchSampler::new(4, 4);
        assert_eq!(sampler.epoch(), 0);
        let _ = sampler.next_batch();
        let _ = sampler.next_batch(); // triggers epoch wrap
        assert_eq!(sampler.epoch(), 1);
    }

    #[test]
    fn mini_batch_shuffle_changes_order() {
        let mut s1 = MiniBatchSampler::new(20, 20).with_shuffle(123);
        let mut s2 = MiniBatchSampler::new(20, 20); // no shuffle
        let b1 = s1.next_batch().to_vec();
        let b2 = s2.next_batch().to_vec();
        // Shuffled should differ from sequential
        assert_ne!(b1, b2);
    }

    #[test]
    fn mini_batch_batches_per_epoch() {
        let sampler = MiniBatchSampler::new(10, 3);
        assert_eq!(sampler.batches_per_epoch(), 4); // ceil(10/3)
    }

    #[test]
    fn gradient_accumulator_basic() {
        let mut acc = GradientAccumulator::new(3, 2);
        let ready1 = acc.accumulate(&[1.0, 2.0, 3.0]);
        assert!(!ready1);
        let ready2 = acc.accumulate(&[3.0, 4.0, 5.0]);
        assert!(ready2);
        let avg = acc.take_averaged();
        assert!((avg[0] - 2.0).abs() < 1e-10); // (1+3)/2
        assert!((avg[1] - 3.0).abs() < 1e-10);
        assert!((avg[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn gradient_accumulator_resets_after_take() {
        let mut acc = GradientAccumulator::new(2, 1);
        acc.accumulate(&[5.0, 6.0]);
        let _ = acc.take_averaged();
        assert_eq!(acc.accumulated_steps(), 0);
    }

    #[test]
    fn gradient_accumulator_display() {
        let acc = GradientAccumulator::new(10, 4);
        let s = format!("{acc}");
        assert!(s.contains("0/4"));
    }

    #[test]
    fn sgd_with_clip_integration() {
        let mut opt = SgdOptimizer::new(0.1).with_clip(ClipStrategy::ByValue(0.5));
        let mut params = vec![0.0, 0.0];
        let mut grads = vec![10.0, -10.0];
        opt.step(&mut params, &mut grads);
        // Grads clipped to [-0.5, 0.5], then step
        assert!((params[0] - (-0.05)).abs() < 1e-10);
        assert!((params[1] - 0.05).abs() < 1e-10);
    }
}
