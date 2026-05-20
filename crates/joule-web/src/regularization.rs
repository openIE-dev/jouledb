//! Regularization techniques for preventing neural network overfitting.
//!
//! Adds penalty terms to the loss function or modifies gradients to
//! encourage simpler models and improve generalization:
//!
//! - [`L1Regularizer`] — Lasso: promotes sparsity via absolute value penalty
//! - [`L2Regularizer`] — Ridge: penalizes large weights via squared magnitude
//! - [`ElasticNetRegularizer`] — combination of L1 and L2
//! - [`WeightDecay`] — direct parameter shrinkage (decoupled from gradient)
//! - [`GradientPenalty`] — constrains gradient norms for Lipschitz continuity
//! - [`SpectralNorm`] — constrains the spectral norm (largest singular value)
//! - [`DropoutMask`] — random zeroing of activations during training

use std::fmt;

// ── L1 Regularization ──────────────────────────────────────────────

/// L1 (Lasso) regularization: `R(θ) = λ * Σ |θ_i|`
///
/// Promotes sparsity — drives small weights exactly to zero.
/// The gradient of L1 is the sign function (subgradient at zero).
pub struct L1Regularizer {
    lambda: f64,
}

impl L1Regularizer {
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda: lambda.abs(),
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda.abs();
        self
    }

    /// Compute the L1 penalty for a weight vector.
    pub fn penalty(&self, weights: &[f64]) -> f64 {
        self.lambda * weights.iter().map(|w| w.abs()).sum::<f64>()
    }

    /// Add L1 gradient contribution to existing gradients.
    /// `dR/dw_i = λ * sign(w_i)`
    pub fn add_gradient(&self, weights: &[f64], grads: &mut [f64]) {
        assert_eq!(weights.len(), grads.len());
        for (g, w) in grads.iter_mut().zip(weights.iter()) {
            *g += self.lambda * w.signum();
        }
    }

    /// Proximal operator for L1 (soft thresholding).
    /// Used in proximal gradient methods for exact sparsity.
    pub fn proximal(&self, weights: &mut [f64], step_size: f64) {
        let threshold = self.lambda * step_size;
        for w in weights.iter_mut() {
            if *w > threshold {
                *w -= threshold;
            } else if *w < -threshold {
                *w += threshold;
            } else {
                *w = 0.0;
            }
        }
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Count weights that are effectively zero.
    pub fn sparsity(&self, weights: &[f64]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }
        let zeros = weights.iter().filter(|w| w.abs() < 1e-10).count();
        zeros as f64 / weights.len() as f64
    }
}

impl fmt::Display for L1Regularizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L1(λ={:.6})", self.lambda)
    }
}

// ── L2 Regularization ──────────────────────────────────────────────

/// L2 (Ridge) regularization: `R(θ) = (λ/2) * Σ θ_i²`
///
/// Penalizes large weights but does not drive them to zero.
/// Results in smoother, more distributed weight distributions.
pub struct L2Regularizer {
    lambda: f64,
}

impl L2Regularizer {
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda: lambda.abs(),
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda.abs();
        self
    }

    /// Compute the L2 penalty for a weight vector.
    pub fn penalty(&self, weights: &[f64]) -> f64 {
        0.5 * self.lambda * weights.iter().map(|w| w * w).sum::<f64>()
    }

    /// Add L2 gradient contribution to existing gradients.
    /// `dR/dw_i = λ * w_i`
    pub fn add_gradient(&self, weights: &[f64], grads: &mut [f64]) {
        assert_eq!(weights.len(), grads.len());
        for (g, w) in grads.iter_mut().zip(weights.iter()) {
            *g += self.lambda * w;
        }
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Weight norm: `√(Σ w_i²)`
    pub fn weight_norm(&self, weights: &[f64]) -> f64 {
        weights.iter().map(|w| w * w).sum::<f64>().sqrt()
    }
}

impl fmt::Display for L2Regularizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L2(λ={:.6})", self.lambda)
    }
}

// ── Elastic Net ────────────────────────────────────────────────────

/// Elastic Net: combines L1 and L2 regularization.
///
/// ```text
/// R(θ) = α * λ * Σ|θ_i| + (1 - α) * (λ/2) * Σ θ_i²
/// ```
///
/// The mixing ratio `α` controls the balance (α=1 is pure L1, α=0 is pure L2).
pub struct ElasticNetRegularizer {
    lambda: f64,
    alpha: f64,
    l1: L1Regularizer,
    l2: L2Regularizer,
}

impl ElasticNetRegularizer {
    pub fn new(lambda: f64, alpha: f64) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self {
            lambda: lambda.abs(),
            alpha,
            l1: L1Regularizer::new(alpha * lambda.abs()),
            l2: L2Regularizer::new((1.0 - alpha) * lambda.abs()),
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self.l1 = L1Regularizer::new(self.alpha * self.lambda);
        self.l2 = L2Regularizer::new((1.0 - self.alpha) * self.lambda);
        self
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda.abs();
        self.l1 = L1Regularizer::new(self.alpha * self.lambda);
        self.l2 = L2Regularizer::new((1.0 - self.alpha) * self.lambda);
        self
    }

    pub fn penalty(&self, weights: &[f64]) -> f64 {
        self.l1.penalty(weights) + self.l2.penalty(weights)
    }

    pub fn add_gradient(&self, weights: &[f64], grads: &mut [f64]) {
        self.l1.add_gradient(weights, grads);
        self.l2.add_gradient(weights, grads);
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl fmt::Display for ElasticNetRegularizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ElasticNet(λ={:.6}, α={:.4})",
            self.lambda, self.alpha
        )
    }
}

// ── Weight Decay ───────────────────────────────────────────────────

/// Decoupled weight decay — shrinks parameters directly rather than
/// adding to the gradient. This is the approach used in AdamW.
///
/// ```text
/// θ_t = (1 - lr * decay) * θ_{t-1}
/// ```
pub struct WeightDecay {
    decay_rate: f64,
    exclude_bias: bool,
    total_steps: u64,
}

impl WeightDecay {
    pub fn new(decay_rate: f64) -> Self {
        Self {
            decay_rate: decay_rate.abs(),
            exclude_bias: false,
            total_steps: 0,
        }
    }

    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate.abs();
        self
    }

    /// When true, bias parameters (identified by caller) are not decayed.
    pub fn with_exclude_bias(mut self, exclude: bool) -> Self {
        self.exclude_bias = exclude;
        self
    }

    /// Apply weight decay in-place.
    pub fn apply(&mut self, params: &mut [f64], learning_rate: f64) {
        let factor = 1.0 - learning_rate * self.decay_rate;
        for p in params.iter_mut() {
            *p *= factor;
        }
        self.total_steps += 1;
    }

    /// Apply decay selectively — `is_bias[i]` marks bias parameters.
    pub fn apply_selective(
        &mut self,
        params: &mut [f64],
        is_bias: &[bool],
        learning_rate: f64,
    ) {
        assert_eq!(params.len(), is_bias.len());
        let factor = 1.0 - learning_rate * self.decay_rate;
        for (p, &bias) in params.iter_mut().zip(is_bias.iter()) {
            if !(self.exclude_bias && bias) {
                *p *= factor;
            }
        }
        self.total_steps += 1;
    }

    pub fn decay_rate(&self) -> f64 {
        self.decay_rate
    }

    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Effective shrinkage factor after `n` steps.
    pub fn effective_shrinkage(&self, n: u64, learning_rate: f64) -> f64 {
        (1.0 - learning_rate * self.decay_rate).powi(n as i32)
    }
}

impl fmt::Display for WeightDecay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WeightDecay(rate={:.6}, steps={})",
            self.decay_rate, self.total_steps
        )
    }
}

// ── Gradient Penalty ───────────────────────────────────────────────

/// Gradient penalty for enforcing Lipschitz continuity (as in WGAN-GP).
///
/// Penalizes deviations of the gradient norm from a target value:
/// ```text
/// GP = λ * (‖∇D(x̃)‖₂ - target)²
/// ```
pub struct GradientPenalty {
    lambda: f64,
    target_norm: f64,
}

impl GradientPenalty {
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            target_norm: 1.0,
        }
    }

    pub fn with_target_norm(mut self, target: f64) -> Self {
        self.target_norm = target;
        self
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Compute the gradient penalty given the gradient vector.
    pub fn compute(&self, gradients: &[f64]) -> f64 {
        let norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        self.lambda * (norm - self.target_norm).powi(2)
    }

    /// Compute penalty gradient w.r.t. the input gradients.
    pub fn penalty_gradient(&self, gradients: &[f64]) -> Vec<f64> {
        let norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return vec![0.0; gradients.len()];
        }
        let coeff = 2.0 * self.lambda * (norm - self.target_norm) / norm;
        gradients.iter().map(|g| coeff * g).collect()
    }
}

impl fmt::Display for GradientPenalty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradientPenalty(λ={:.6}, target={:.4})",
            self.lambda, self.target_norm
        )
    }
}

// ── Spectral Norm ──────────────────────────────────────────────────

/// Spectral normalization constrains the largest singular value of a
/// weight matrix to be at most `max_sigma`.
///
/// Uses power iteration to approximate the spectral norm, then rescales
/// the weight matrix: `W_sn = W / max(1, σ₁(W) / max_sigma)`
pub struct SpectralNorm {
    max_sigma: f64,
    power_iterations: usize,
    u: Vec<f64>,
    v: Vec<f64>,
}

impl SpectralNorm {
    pub fn new(max_sigma: f64) -> Self {
        Self {
            max_sigma: max_sigma.abs().max(1e-10),
            power_iterations: 1,
            u: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn with_power_iterations(mut self, n: usize) -> Self {
        self.power_iterations = n.max(1);
        self
    }

    pub fn with_max_sigma(mut self, sigma: f64) -> Self {
        self.max_sigma = sigma.abs().max(1e-10);
        self
    }

    /// Estimate the spectral norm of a matrix (rows × cols, row-major)
    /// using power iteration, and normalize the matrix in-place.
    pub fn normalize(&mut self, weights: &mut [f64], rows: usize, cols: usize) -> f64 {
        assert_eq!(weights.len(), rows * cols);

        // Initialize u, v if needed
        if self.u.len() != rows {
            self.u = vec![1.0 / (rows as f64).sqrt(); rows];
        }
        if self.v.len() != cols {
            self.v = vec![1.0 / (cols as f64).sqrt(); cols];
        }

        // Power iteration to approximate σ₁
        for _ in 0..self.power_iterations {
            // v = W^T u / ‖W^T u‖
            for j in 0..cols {
                let mut s = 0.0;
                for i in 0..rows {
                    s += weights[i * cols + j] * self.u[i];
                }
                self.v[j] = s;
            }
            let v_norm = self.v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if v_norm > 1e-15 {
                for x in self.v.iter_mut() {
                    *x /= v_norm;
                }
            }

            // u = W v / ‖W v‖
            for i in 0..rows {
                let mut s = 0.0;
                for j in 0..cols {
                    s += weights[i * cols + j] * self.v[j];
                }
                self.u[i] = s;
            }
            let u_norm = self.u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if u_norm > 1e-15 {
                for x in self.u.iter_mut() {
                    *x /= u_norm;
                }
            }
        }

        // σ₁ ≈ u^T W v
        let mut sigma = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                sigma += self.u[i] * weights[i * cols + j] * self.v[j];
            }
        }
        sigma = sigma.abs();

        // Normalize if exceeding max_sigma
        if sigma > self.max_sigma {
            let scale = self.max_sigma / sigma;
            for w in weights.iter_mut() {
                *w *= scale;
            }
        }

        sigma
    }

    pub fn max_sigma(&self) -> f64 {
        self.max_sigma
    }
}

impl fmt::Display for SpectralNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpectralNorm(σ_max={:.4}, iters={})",
            self.max_sigma, self.power_iterations
        )
    }
}

// ── Dropout Mask ───────────────────────────────────────────────────

/// Generates a binary dropout mask for zeroing random activations.
///
/// During training, each activation is independently zeroed with
/// probability `p`, and surviving values are scaled by `1/(1-p)`
/// (inverted dropout) to maintain expected values.
pub struct DropoutMask {
    drop_prob: f64,
    rng_state: u64,
    seed: u64,
}

impl DropoutMask {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            drop_prob: drop_prob.clamp(0.0, 1.0),
            rng_state: 42,
            seed: 42,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng_state = seed;
        self
    }

    pub fn with_drop_prob(mut self, prob: f64) -> Self {
        self.drop_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Generate a mask vector of length `n`. Values are either `1/(1-p)` or `0`.
    pub fn generate_mask(&mut self, n: usize) -> Vec<f64> {
        let scale = if self.drop_prob < 1.0 {
            1.0 / (1.0 - self.drop_prob)
        } else {
            0.0
        };
        (0..n)
            .map(|_| {
                let r = self.next_f64();
                if r < self.drop_prob {
                    0.0
                } else {
                    scale
                }
            })
            .collect()
    }

    /// Apply dropout in-place to an activation vector.
    pub fn apply(&mut self, activations: &mut [f64]) {
        let mask = self.generate_mask(activations.len());
        for (a, m) in activations.iter_mut().zip(mask.iter()) {
            *a *= m;
        }
    }

    pub fn drop_prob(&self) -> f64 {
        self.drop_prob
    }

    fn next_f64(&mut self) -> f64 {
        self.rng_state = self.rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 33) as f64) / ((1u64 << 31) as f64)
    }
}

impl fmt::Display for DropoutMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dropout(p={:.4})", self.drop_prob)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_penalty() {
        let reg = L1Regularizer::new(0.1);
        let weights = vec![1.0, -2.0, 3.0];
        // 0.1 * (1 + 2 + 3) = 0.6
        assert!((reg.penalty(&weights) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn l1_gradient() {
        let reg = L1Regularizer::new(0.5);
        let weights = vec![2.0, -1.0, 0.0];
        let mut grads = vec![0.0, 0.0, 0.0];
        reg.add_gradient(&weights, &mut grads);
        assert!((grads[0] - 0.5).abs() < 1e-10);  // sign(2) * 0.5
        assert!((grads[1] - (-0.5)).abs() < 1e-10); // sign(-1) * 0.5
    }

    #[test]
    fn l1_proximal_sparsity() {
        let reg = L1Regularizer::new(1.0);
        let mut weights = vec![0.5, -0.3, 2.0, -1.5];
        reg.proximal(&mut weights, 0.4); // threshold = 0.4
        assert!((weights[0] - 0.1).abs() < 1e-10);
        assert!((weights[1] - 0.0).abs() < 1e-10); // thresholded to zero
    }

    #[test]
    fn l2_penalty() {
        let reg = L2Regularizer::new(0.1);
        let weights = vec![3.0, 4.0];
        // 0.5 * 0.1 * (9 + 16) = 1.25
        assert!((reg.penalty(&weights) - 1.25).abs() < 1e-10);
    }

    #[test]
    fn l2_gradient() {
        let reg = L2Regularizer::new(0.2);
        let weights = vec![5.0];
        let mut grads = vec![0.0];
        reg.add_gradient(&weights, &mut grads);
        // 0.2 * 5 = 1.0
        assert!((grads[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn l2_weight_norm() {
        let reg = L2Regularizer::new(0.1);
        let weights = vec![3.0, 4.0];
        assert!((reg.weight_norm(&weights) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn elastic_net_combines() {
        let l1_only = ElasticNetRegularizer::new(0.1, 1.0);
        let l2_only = ElasticNetRegularizer::new(0.1, 0.0);
        let mixed = ElasticNetRegularizer::new(0.1, 0.5);
        let weights = vec![1.0, -2.0];
        let p1 = l1_only.penalty(&weights);
        let p2 = l2_only.penalty(&weights);
        let pm = mixed.penalty(&weights);
        // Mixed should be between pure L1 and pure L2
        assert!(pm > 0.0);
        assert!((p1 - L1Regularizer::new(0.1).penalty(&weights)).abs() < 1e-10);
        assert!((p2 - L2Regularizer::new(0.1).penalty(&weights)).abs() < 1e-10);
    }

    #[test]
    fn elastic_net_display() {
        let reg = ElasticNetRegularizer::new(0.01, 0.5);
        assert!(format!("{reg}").contains("ElasticNet"));
    }

    #[test]
    fn weight_decay_shrinks() {
        let mut wd = WeightDecay::new(0.01);
        let mut params = vec![10.0, -5.0];
        wd.apply(&mut params, 0.1);
        // factor = 1 - 0.1*0.01 = 0.999
        assert!((params[0] - 9.99).abs() < 1e-10);
    }

    #[test]
    fn weight_decay_selective() {
        let mut wd = WeightDecay::new(0.1).with_exclude_bias(true);
        let mut params = vec![10.0, 5.0]; // [weight, bias]
        let is_bias = vec![false, true];
        wd.apply_selective(&mut params, &is_bias, 0.1);
        assert!(params[0] < 10.0); // decayed
        assert!((params[1] - 5.0).abs() < 1e-10); // bias excluded
    }

    #[test]
    fn weight_decay_effective_shrinkage() {
        let wd = WeightDecay::new(0.01);
        let shrink = wd.effective_shrinkage(100, 0.1);
        // (1 - 0.001)^100 ≈ 0.9048
        assert!((shrink - 0.9048).abs() < 0.01);
    }

    #[test]
    fn gradient_penalty_at_target() {
        let gp = GradientPenalty::new(10.0);
        let grads = vec![0.6, 0.8]; // norm = 1.0 = target
        let penalty = gp.compute(&grads);
        assert!(penalty.abs() < 1e-10);
    }

    #[test]
    fn gradient_penalty_away_from_target() {
        let gp = GradientPenalty::new(10.0);
        let grads = vec![1.5, 2.0]; // norm = 2.5
        let penalty = gp.compute(&grads);
        // 10 * (2.5 - 1.0)^2 = 10 * 2.25 = 22.5
        assert!((penalty - 22.5).abs() < 1e-10);
    }

    #[test]
    fn spectral_norm_basic() {
        let mut sn = SpectralNorm::new(1.0).with_power_iterations(10);
        // Identity matrix: σ₁ = 1.0
        let mut weights = vec![1.0, 0.0, 0.0, 1.0];
        let sigma = sn.normalize(&mut weights, 2, 2);
        assert!((sigma - 1.0).abs() < 0.1);
    }

    #[test]
    fn spectral_norm_constrains() {
        let mut sn = SpectralNorm::new(1.0).with_power_iterations(20);
        // Scaled matrix: σ₁ ≈ 10
        let mut weights = vec![10.0, 0.0, 0.0, 10.0];
        sn.normalize(&mut weights, 2, 2);
        // After normalization, max element should be ≈ 1.0
        let max_w = weights.iter().map(|w| w.abs()).fold(0.0_f64, f64::max);
        assert!(max_w <= 1.5);
    }

    #[test]
    fn spectral_norm_display() {
        let sn = SpectralNorm::new(1.0);
        assert!(format!("{sn}").contains("SpectralNorm"));
    }

    #[test]
    fn dropout_zeroes_some() {
        let mut dropout = DropoutMask::new(0.5).with_seed(42);
        let mut activations = vec![1.0; 100];
        dropout.apply(&mut activations);
        let zeros = activations.iter().filter(|a| **a == 0.0).count();
        // Roughly 50% should be zero
        assert!(zeros > 20 && zeros < 80);
    }

    #[test]
    fn dropout_zero_prob_keeps_all() {
        let mut dropout = DropoutMask::new(0.0);
        let mask = dropout.generate_mask(10);
        assert!(mask.iter().all(|m| (m - 1.0).abs() < 1e-10));
    }

    #[test]
    fn dropout_display() {
        let d = DropoutMask::new(0.3);
        assert!(format!("{d}").contains("Dropout"));
    }

    #[test]
    fn l1_display() {
        let r = L1Regularizer::new(0.01);
        assert!(format!("{r}").contains("L1"));
    }
}
