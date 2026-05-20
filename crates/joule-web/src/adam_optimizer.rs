//! Adam-family optimizers for neural network training.
//!
//! Implements adaptive learning rate methods that maintain per-parameter
//! first and second moment estimates:
//!
//! - [`AdamOptimizer`] — classic Adam with bias correction
//! - [`AdamWOptimizer`] — decoupled weight decay regularization
//! - [`AMSGradOptimizer`] — AMSGrad variant with max second-moment tracking
//! - [`AdamConfig`] — shared hyperparameter configuration

use std::fmt;

// ── Configuration ──────────────────────────────────────────────────

/// Shared hyperparameters for Adam-family optimizers.
#[derive(Debug, Clone)]
pub struct AdamConfig {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
}

impl AdamConfig {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self::new(0.001)
    }
}

impl fmt::Display for AdamConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AdamConfig(lr={:.6}, β1={:.4}, β2={:.4}, ε={:.1e}, wd={:.6})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay
        )
    }
}

// ── Adam Optimizer State ───────────────────────────────────────────

/// Per-parameter optimizer state for Adam variants.
struct AdamState {
    m: Vec<f64>,     // first moment (mean of gradients)
    v: Vec<f64>,     // second moment (mean of squared gradients)
    step: u64,
}

impl AdamState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
        }
    }

    fn ensure_size(&mut self, size: usize) {
        if self.m.len() != size {
            self.m = vec![0.0; size];
            self.v = vec![0.0; size];
            self.step = 0;
        }
    }
}

// ── Classic Adam ───────────────────────────────────────────────────

/// Adam optimizer (Kingma & Ba, 2014).
///
/// Maintains exponential moving averages of the gradient (first moment)
/// and squared gradient (second moment), with bias correction:
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * g_t
/// v_t = β2 * v_{t-1} + (1 - β2) * g_t²
/// m̂_t = m_t / (1 - β1^t)
/// v̂_t = v_t / (1 - β2^t)
/// θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
/// ```
pub struct AdamOptimizer {
    config: AdamConfig,
    state: AdamState,
}

impl AdamOptimizer {
    pub fn new(config: AdamConfig) -> Self {
        Self {
            config,
            state: AdamState::new(0),
        }
    }

    pub fn from_lr(learning_rate: f64) -> Self {
        Self::new(AdamConfig::new(learning_rate))
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.config.beta1 = beta1;
        self.config.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Perform a single parameter update.
    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        self.state.ensure_size(params.len());
        self.state.step += 1;

        let t = self.state.step as f64;
        let bias_correction1 = 1.0 - self.config.beta1.powf(t);
        let bias_correction2 = 1.0 - self.config.beta2.powf(t);
        let lr = self.config.learning_rate;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let eps = self.config.epsilon;

        for i in 0..params.len() {
            // L2 regularization (classic Adam applies it to gradient)
            let grad_with_decay = grads[i] + self.config.weight_decay * params[i];

            // Update biased first moment estimate
            self.state.m[i] = b1 * self.state.m[i] + (1.0 - b1) * grad_with_decay;
            // Update biased second raw moment estimate
            self.state.v[i] = b2 * self.state.v[i] + (1.0 - b2) * grad_with_decay * grad_with_decay;

            // Bias-corrected estimates
            let m_hat = self.state.m[i] / bias_correction1;
            let v_hat = self.state.v[i] / bias_correction2;

            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    pub fn step_count(&self) -> u64 {
        self.state.step
    }

    pub fn learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    pub fn config(&self) -> &AdamConfig {
        &self.config
    }

    pub fn first_moment(&self) -> &[f64] {
        &self.state.m
    }

    pub fn second_moment(&self) -> &[f64] {
        &self.state.v
    }

    pub fn reset(&mut self) {
        self.state = AdamState::new(0);
    }
}

impl fmt::Display for AdamOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Adam(lr={:.6}, β1={:.4}, β2={:.4}, steps={})",
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.state.step
        )
    }
}

// ── AdamW (Decoupled Weight Decay) ─────────────────────────────────

/// AdamW optimizer (Loshchilov & Hutter, 2017).
///
/// Unlike classic Adam, weight decay is applied directly to the parameters
/// rather than being added to the gradient. This decoupling improves
/// regularization when used with adaptive learning rates.
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * g_t
/// v_t = β2 * v_{t-1} + (1 - β2) * g_t²
/// m̂_t = m_t / (1 - β1^t)
/// v̂_t = v_t / (1 - β2^t)
/// θ_t = (1 - lr * λ) * θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
/// ```
pub struct AdamWOptimizer {
    config: AdamConfig,
    state: AdamState,
}

impl AdamWOptimizer {
    pub fn new(config: AdamConfig) -> Self {
        Self {
            config,
            state: AdamState::new(0),
        }
    }

    pub fn from_lr(learning_rate: f64) -> Self {
        Self::new(AdamConfig::new(learning_rate).with_weight_decay(0.01))
    }

    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.config.weight_decay = wd;
        self
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.config.beta1 = beta1;
        self.config.beta2 = beta2;
        self
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        self.state.ensure_size(params.len());
        self.state.step += 1;

        let t = self.state.step as f64;
        let bias_correction1 = 1.0 - self.config.beta1.powf(t);
        let bias_correction2 = 1.0 - self.config.beta2.powf(t);
        let lr = self.config.learning_rate;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let eps = self.config.epsilon;
        let wd = self.config.weight_decay;

        for i in 0..params.len() {
            // Decoupled weight decay: applied to params, NOT gradient
            params[i] *= 1.0 - lr * wd;

            // Moment updates use raw gradient (no weight decay)
            self.state.m[i] = b1 * self.state.m[i] + (1.0 - b1) * grads[i];
            self.state.v[i] = b2 * self.state.v[i] + (1.0 - b2) * grads[i] * grads[i];

            let m_hat = self.state.m[i] / bias_correction1;
            let v_hat = self.state.v[i] / bias_correction2;

            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    pub fn step_count(&self) -> u64 {
        self.state.step
    }

    pub fn config(&self) -> &AdamConfig {
        &self.config
    }

    pub fn reset(&mut self) {
        self.state = AdamState::new(0);
    }
}

impl fmt::Display for AdamWOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AdamW(lr={:.6}, wd={:.6}, β1={:.4}, β2={:.4}, steps={})",
            self.config.learning_rate,
            self.config.weight_decay,
            self.config.beta1,
            self.config.beta2,
            self.state.step
        )
    }
}

// ── AMSGrad ────────────────────────────────────────────────────────

/// AMSGrad variant of Adam (Reddi, Kale & Kumar, 2018).
///
/// Maintains a running maximum of the second moment estimate to ensure
/// non-increasing step sizes, which fixes a convergence issue in vanilla Adam.
///
/// ```text
/// v̂_t = max(v̂_{t-1}, v_t / (1 - β2^t))
/// θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
/// ```
pub struct AMSGradOptimizer {
    config: AdamConfig,
    state: AdamState,
    v_max: Vec<f64>,
}

impl AMSGradOptimizer {
    pub fn new(config: AdamConfig) -> Self {
        Self {
            config,
            state: AdamState::new(0),
            v_max: Vec::new(),
        }
    }

    pub fn from_lr(learning_rate: f64) -> Self {
        Self::new(AdamConfig::new(learning_rate))
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.config.beta1 = beta1;
        self.config.beta2 = beta2;
        self
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        self.state.ensure_size(params.len());
        if self.v_max.len() != params.len() {
            self.v_max = vec![0.0; params.len()];
        }
        self.state.step += 1;

        let t = self.state.step as f64;
        let bias_correction1 = 1.0 - self.config.beta1.powf(t);
        let bias_correction2 = 1.0 - self.config.beta2.powf(t);
        let lr = self.config.learning_rate;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let eps = self.config.epsilon;

        for i in 0..params.len() {
            self.state.m[i] = b1 * self.state.m[i] + (1.0 - b1) * grads[i];
            self.state.v[i] = b2 * self.state.v[i] + (1.0 - b2) * grads[i] * grads[i];

            let m_hat = self.state.m[i] / bias_correction1;
            let v_hat = self.state.v[i] / bias_correction2;

            // AMSGrad: take the max of all past v_hat values
            if v_hat > self.v_max[i] {
                self.v_max[i] = v_hat;
            }

            params[i] -= lr * m_hat / (self.v_max[i].sqrt() + eps);
        }
    }

    pub fn step_count(&self) -> u64 {
        self.state.step
    }

    pub fn v_max(&self) -> &[f64] {
        &self.v_max
    }

    pub fn config(&self) -> &AdamConfig {
        &self.config
    }

    pub fn reset(&mut self) {
        self.state = AdamState::new(0);
        self.v_max.clear();
    }
}

impl fmt::Display for AMSGradOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AMSGrad(lr={:.6}, β1={:.4}, β2={:.4}, steps={})",
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.state.step
        )
    }
}

// ── Utility: Gradient Statistics ───────────────────────────────────

/// Summary statistics for a gradient vector, useful for monitoring training.
#[derive(Debug, Clone)]
pub struct GradientStats {
    pub mean: f64,
    pub variance: f64,
    pub l2_norm: f64,
    pub max_abs: f64,
    pub min_abs: f64,
    pub count: usize,
}

impl GradientStats {
    pub fn compute(grads: &[f64]) -> Self {
        if grads.is_empty() {
            return Self {
                mean: 0.0,
                variance: 0.0,
                l2_norm: 0.0,
                max_abs: 0.0,
                min_abs: 0.0,
                count: 0,
            };
        }
        let n = grads.len() as f64;
        let sum: f64 = grads.iter().sum();
        let mean = sum / n;
        let variance = grads.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / n;
        let l2_norm = grads.iter().map(|g| g * g).sum::<f64>().sqrt();
        let max_abs = grads.iter().map(|g| g.abs()).fold(0.0_f64, f64::max);
        let min_abs = grads.iter().map(|g| g.abs()).fold(f64::INFINITY, f64::min);
        Self {
            mean,
            variance,
            l2_norm,
            max_abs,
            min_abs,
            count: grads.len(),
        }
    }
}

impl fmt::Display for GradientStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradStats(μ={:.4e}, σ²={:.4e}, ‖g‖={:.4e}, n={})",
            self.mean, self.variance, self.l2_norm, self.count
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_config_defaults() {
        let cfg = AdamConfig::default();
        assert!((cfg.learning_rate - 0.001).abs() < 1e-10);
        assert!((cfg.beta1 - 0.9).abs() < 1e-10);
        assert!((cfg.beta2 - 0.999).abs() < 1e-10);
    }

    #[test]
    fn adam_config_builder() {
        let cfg = AdamConfig::new(0.01).with_betas(0.8, 0.99).with_epsilon(1e-7);
        assert!((cfg.beta1 - 0.8).abs() < 1e-10);
        assert!((cfg.epsilon - 1e-7).abs() < 1e-15);
    }

    #[test]
    fn adam_basic_step() {
        let mut opt = AdamOptimizer::from_lr(0.01);
        let mut params = vec![1.0, 2.0];
        let grads = vec![0.1, 0.2];
        opt.step(&mut params, &grads);
        // Parameters should decrease (positive gradients)
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn adam_bias_correction_matters() {
        let mut opt = AdamOptimizer::from_lr(0.001);
        let mut p = vec![0.0];
        // First step with large gradient — bias correction amplifies update
        opt.step(&mut p, &[10.0]);
        // Without bias correction, the update would be ~0.001
        // With correction at t=1: m_hat = g (since β1^1 correction), similar for v
        assert!(p[0].abs() > 0.0005);
    }

    #[test]
    fn adam_step_count() {
        let mut opt = AdamOptimizer::from_lr(0.001);
        let mut p = vec![0.0];
        for _ in 0..10 {
            opt.step(&mut p, &[1.0]);
        }
        assert_eq!(opt.step_count(), 10);
    }

    #[test]
    fn adam_display() {
        let opt = AdamOptimizer::from_lr(0.001);
        let s = format!("{opt}");
        assert!(s.contains("Adam("));
        assert!(s.contains("0.001"));
    }

    #[test]
    fn adam_reset() {
        let mut opt = AdamOptimizer::from_lr(0.001);
        opt.step(&mut vec![0.0], &[1.0]);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn adamw_decoupled_weight_decay() {
        let mut adam = AdamOptimizer::new(AdamConfig::new(0.01).with_weight_decay(0.1));
        let mut adamw = AdamWOptimizer::new(AdamConfig::new(0.01).with_weight_decay(0.1));
        let mut p_adam = vec![5.0];
        let mut p_adamw = vec![5.0];
        for _ in 0..5 {
            adam.step(&mut p_adam, &[0.1]);
            adamw.step(&mut p_adamw, &[0.1]);
        }
        // They should produce different results due to decoupling
        assert!((p_adam[0] - p_adamw[0]).abs() > 1e-6);
    }

    #[test]
    fn adamw_shrinks_params() {
        let mut opt = AdamWOptimizer::from_lr(0.01).with_weight_decay(0.1);
        let initial = 10.0;
        let mut params = vec![initial];
        opt.step(&mut params, &[0.0]); // zero gradient, only decay
        // Weight decay should shrink the parameter
        assert!(params[0] < initial);
    }

    #[test]
    fn adamw_display() {
        let opt = AdamWOptimizer::from_lr(0.001).with_weight_decay(0.01);
        let s = format!("{opt}");
        assert!(s.contains("AdamW"));
    }

    #[test]
    fn amsgrad_non_increasing_v() {
        let mut opt = AMSGradOptimizer::from_lr(0.001);
        let mut p = vec![0.0];
        // Large gradient first, then small
        opt.step(&mut p, &[10.0]);
        let v1 = opt.v_max()[0];
        opt.step(&mut p, &[0.001]);
        let v2 = opt.v_max()[0];
        // v_max should never decrease
        assert!(v2 >= v1);
    }

    #[test]
    fn amsgrad_differs_from_adam() {
        let mut adam = AdamOptimizer::from_lr(0.01);
        let mut ams = AMSGradOptimizer::from_lr(0.01);
        let mut pa = vec![0.0];
        let mut pb = vec![0.0];
        // Sequence with varying gradient magnitudes
        let grads = [10.0, 0.01, 10.0, 0.01, 10.0];
        for g in &grads {
            adam.step(&mut pa, &[*g]);
            ams.step(&mut pb, &[*g]);
        }
        // AMSGrad is more conservative due to v_max
        assert!((pa[0] - pb[0]).abs() > 1e-8);
    }

    #[test]
    fn amsgrad_display() {
        let opt = AMSGradOptimizer::from_lr(0.001);
        assert!(format!("{opt}").contains("AMSGrad"));
    }

    #[test]
    fn amsgrad_reset() {
        let mut opt = AMSGradOptimizer::from_lr(0.001);
        opt.step(&mut vec![0.0], &[1.0]);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
        assert!(opt.v_max().is_empty());
    }

    #[test]
    fn gradient_stats_basic() {
        let grads = vec![1.0, -1.0, 2.0, -2.0];
        let stats = GradientStats::compute(&grads);
        assert!((stats.mean - 0.0).abs() < 1e-10);
        assert_eq!(stats.count, 4);
        assert!((stats.max_abs - 2.0).abs() < 1e-10);
        assert!((stats.min_abs - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gradient_stats_empty() {
        let stats = GradientStats::compute(&[]);
        assert_eq!(stats.count, 0);
        assert!((stats.l2_norm - 0.0).abs() < 1e-10);
    }

    #[test]
    fn gradient_stats_display() {
        let stats = GradientStats::compute(&[1.0, 2.0, 3.0]);
        let s = format!("{stats}");
        assert!(s.contains("GradStats"));
    }

    #[test]
    fn adam_config_display() {
        let cfg = AdamConfig::default();
        let s = format!("{cfg}");
        assert!(s.contains("AdamConfig"));
    }

    #[test]
    fn adam_moments_populated() {
        let mut opt = AdamOptimizer::from_lr(0.001);
        let mut p = vec![0.0, 0.0];
        opt.step(&mut p, &[1.0, 2.0]);
        assert_eq!(opt.first_moment().len(), 2);
        assert_eq!(opt.second_moment().len(), 2);
        assert!(opt.first_moment()[0] > 0.0);
    }
}
