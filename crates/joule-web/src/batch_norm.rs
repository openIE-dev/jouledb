//! Batch normalization layer for stabilizing neural network training.
//!
//! Normalizes activations across the batch dimension, maintaining
//! running statistics for inference mode. Supports configurable
//! momentum, epsilon, and optional affine parameters (gamma/beta).

use std::fmt;

// ── BatchNormMode ─────────────────────────────────────────────────

/// Whether the layer is in training or inference mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BNMode {
    Training,
    Inference,
}

impl fmt::Display for BNMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Training => write!(f, "train"),
            Self::Inference => write!(f, "eval"),
        }
    }
}

// ── BatchNormConfig ───────────────────────────────────────────────

/// Configuration for batch normalization.
#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    pub num_features: usize,
    pub epsilon: f64,
    pub momentum: f64,
    pub affine: bool,
    pub track_running_stats: bool,
}

impl BatchNormConfig {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            epsilon: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        }
    }

    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_no_affine(mut self) -> Self {
        self.affine = false;
        self
    }

    pub fn with_no_tracking(mut self) -> Self {
        self.track_running_stats = false;
        self
    }
}

// ── BatchNormLayer ────────────────────────────────────────────────

/// Batch normalization layer.
///
/// During training: normalizes using batch statistics.
/// During inference: normalizes using running statistics.
///
/// Output: `y = gamma * (x - mean) / sqrt(var + eps) + beta`
#[derive(Debug, Clone)]
pub struct BatchNormLayer {
    pub config: BatchNormConfig,
    pub mode: BNMode,
    /// Learned scale parameters (gamma).
    pub gamma: Vec<f64>,
    /// Learned shift parameters (beta).
    pub beta: Vec<f64>,
    /// Running mean (exponential moving average).
    pub running_mean: Vec<f64>,
    /// Running variance (exponential moving average).
    pub running_var: Vec<f64>,
    /// Number of batches seen (for running stats).
    pub num_batches_tracked: u64,
    // Cached values for backward pass
    last_normalized: Vec<f64>,
    last_std_inv: Vec<f64>,
    last_batch_mean: Vec<f64>,
    last_input: Vec<f64>,
    last_batch_size: usize,
}

impl BatchNormLayer {
    pub fn new(config: BatchNormConfig) -> Self {
        let n = config.num_features;
        Self {
            gamma: vec![1.0; n],
            beta: vec![0.0; n],
            running_mean: vec![0.0; n],
            running_var: vec![1.0; n],
            num_batches_tracked: 0,
            last_normalized: Vec::new(),
            last_std_inv: Vec::new(),
            last_batch_mean: Vec::new(),
            last_input: Vec::new(),
            last_batch_size: 0,
            mode: BNMode::Training,
            config,
        }
    }

    /// Set mode to training.
    pub fn with_training(mut self) -> Self {
        self.mode = BNMode::Training;
        self
    }

    /// Set mode to inference.
    pub fn with_eval(mut self) -> Self {
        self.mode = BNMode::Inference;
        self
    }

    /// Switch between training and inference mode.
    pub fn set_mode(&mut self, mode: BNMode) {
        self.mode = mode;
    }

    /// Number of trainable parameters.
    pub fn param_count(&self) -> usize {
        if self.config.affine { self.config.num_features * 2 } else { 0 }
    }

    /// Forward pass for 1D input: `[batch_size, num_features]` flattened row-major.
    pub fn forward(&mut self, input: &[f64], batch_size: usize) -> Vec<f64> {
        let n = self.config.num_features;
        assert_eq!(input.len(), batch_size * n, "input size mismatch");

        self.last_input = input.to_vec();
        self.last_batch_size = batch_size;

        let mut output = vec![0.0; batch_size * n];

        match self.mode {
            BNMode::Training => {
                // Compute batch mean and variance for each feature
                let mut batch_mean = vec![0.0; n];
                let mut batch_var = vec![0.0; n];

                for f_idx in 0..n {
                    let mut sum = 0.0;
                    for b in 0..batch_size {
                        sum += input[b * n + f_idx];
                    }
                    batch_mean[f_idx] = sum / batch_size as f64;
                }

                for f_idx in 0..n {
                    let mut var_sum = 0.0;
                    for b in 0..batch_size {
                        let diff = input[b * n + f_idx] - batch_mean[f_idx];
                        var_sum += diff * diff;
                    }
                    batch_var[f_idx] = var_sum / batch_size as f64;
                }

                // Normalize
                let mut normalized = vec![0.0; batch_size * n];
                let mut std_inv = vec![0.0; n];

                for f_idx in 0..n {
                    std_inv[f_idx] = 1.0 / (batch_var[f_idx] + self.config.epsilon).sqrt();
                }

                for b in 0..batch_size {
                    for f_idx in 0..n {
                        let idx = b * n + f_idx;
                        normalized[idx] =
                            (input[idx] - batch_mean[f_idx]) * std_inv[f_idx];

                        if self.config.affine {
                            output[idx] =
                                self.gamma[f_idx] * normalized[idx] + self.beta[f_idx];
                        } else {
                            output[idx] = normalized[idx];
                        }
                    }
                }

                // Update running statistics
                if self.config.track_running_stats {
                    let m = self.config.momentum;
                    for f_idx in 0..n {
                        self.running_mean[f_idx] =
                            (1.0 - m) * self.running_mean[f_idx] + m * batch_mean[f_idx];
                        // Use Bessel's correction for running variance
                        let unbiased_var = if batch_size > 1 {
                            batch_var[f_idx] * batch_size as f64 / (batch_size - 1) as f64
                        } else {
                            batch_var[f_idx]
                        };
                        self.running_var[f_idx] =
                            (1.0 - m) * self.running_var[f_idx] + m * unbiased_var;
                    }
                    self.num_batches_tracked += 1;
                }

                self.last_normalized = normalized;
                self.last_std_inv = std_inv;
                self.last_batch_mean = batch_mean;
            }
            BNMode::Inference => {
                // Use running statistics
                for b in 0..batch_size {
                    for f_idx in 0..n {
                        let idx = b * n + f_idx;
                        let normalized = (input[idx] - self.running_mean[f_idx])
                            / (self.running_var[f_idx] + self.config.epsilon).sqrt();
                        if self.config.affine {
                            output[idx] = self.gamma[f_idx] * normalized + self.beta[f_idx];
                        } else {
                            output[idx] = normalized;
                        }
                    }
                }
            }
        }

        output
    }

    /// Forward pass for 2D spatial input: `[batch, channels, H, W]` flattened.
    /// Normalizes per-channel across batch and spatial dimensions.
    pub fn forward_2d(
        &mut self,
        input: &[f64],
        batch_size: usize,
        height: usize,
        width: usize,
    ) -> Vec<f64> {
        let c = self.config.num_features;
        let spatial = height * width;
        assert_eq!(input.len(), batch_size * c * spatial);

        // Reshape to treat spatial dims as part of the "batch" for each channel
        let effective_batch = batch_size * spatial;
        let mut reshaped = vec![0.0; effective_batch * c];

        for b in 0..batch_size {
            for ch in 0..c {
                for s in 0..spatial {
                    let src = b * c * spatial + ch * spatial + s;
                    let dst = (b * spatial + s) * c + ch;
                    reshaped[dst] = input[src];
                }
            }
        }

        let norm_out = self.forward(&reshaped, effective_batch);

        // Reshape back
        let mut output = vec![0.0; input.len()];
        for b in 0..batch_size {
            for ch in 0..c {
                for s in 0..spatial {
                    let src = (b * spatial + s) * c + ch;
                    let dst = b * c * spatial + ch * spatial + s;
                    output[dst] = norm_out[src];
                }
            }
        }

        output
    }

    /// Backward pass: compute gradients for gamma, beta, and input.
    pub fn backward(
        &self,
        grad_output: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.config.num_features;
        let batch_size = self.last_batch_size;
        let bs_f = batch_size as f64;

        let mut grad_gamma = vec![0.0; n];
        let mut grad_beta = vec![0.0; n];
        let mut grad_input = vec![0.0; batch_size * n];

        // Compute grad_gamma and grad_beta
        for f_idx in 0..n {
            for b in 0..batch_size {
                let idx = b * n + f_idx;
                grad_gamma[f_idx] += grad_output[idx] * self.last_normalized[idx];
                grad_beta[f_idx] += grad_output[idx];
            }
        }

        // Compute grad_input
        for f_idx in 0..n {
            let gamma_val = if self.config.affine { self.gamma[f_idx] } else { 1.0 };
            let sinv = self.last_std_inv[f_idx];

            let mut sum_grad = 0.0;
            let mut sum_grad_norm = 0.0;
            for b in 0..batch_size {
                let idx = b * n + f_idx;
                let dxhat = grad_output[idx] * gamma_val;
                sum_grad += dxhat;
                sum_grad_norm += dxhat * self.last_normalized[idx];
            }

            for b in 0..batch_size {
                let idx = b * n + f_idx;
                let dxhat = grad_output[idx] * gamma_val;
                grad_input[idx] = sinv / bs_f
                    * (bs_f * dxhat - sum_grad - self.last_normalized[idx] * sum_grad_norm);
            }
        }

        (grad_input, grad_gamma, grad_beta)
    }

    /// Fuse batch norm into a preceding linear layer's weights/biases for inference.
    /// Returns (fused_weight_scale, fused_bias_offset) per feature.
    pub fn fuse_params(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.config.num_features;
        let mut scale = vec![0.0; n];
        let mut offset = vec![0.0; n];

        for i in 0..n {
            let std_val = (self.running_var[i] + self.config.epsilon).sqrt();
            let g = if self.config.affine { self.gamma[i] } else { 1.0 };
            let b = if self.config.affine { self.beta[i] } else { 0.0 };

            scale[i] = g / std_val;
            offset[i] = b - g * self.running_mean[i] / std_val;
        }

        (scale, offset)
    }
}

impl fmt::Display for BatchNormLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatchNorm(features={}, eps={:.0e}, momentum={}, affine={}, mode={})",
            self.config.num_features,
            self.config.epsilon,
            self.config.momentum,
            self.config.affine,
            self.mode
        )
    }
}

// ── Layer Normalization ───────────────────────────────────────────

/// Layer normalization — normalizes across features (not batch).
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub num_features: usize,
    pub epsilon: f64,
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
}

impl LayerNorm {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            epsilon: 1e-5,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
        }
    }

    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Forward: normalize each sample independently across features.
    pub fn forward(&self, input: &[f64], batch_size: usize) -> Vec<f64> {
        let n = self.num_features;
        assert_eq!(input.len(), batch_size * n);
        let mut output = vec![0.0; batch_size * n];

        for b in 0..batch_size {
            let offset = b * n;
            let slice = &input[offset..offset + n];

            let mean = slice.iter().sum::<f64>() / n as f64;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let std_inv = 1.0 / (var + self.epsilon).sqrt();

            for i in 0..n {
                output[offset + i] = self.gamma[i] * (slice[i] - mean) * std_inv + self.beta[i];
            }
        }

        output
    }
}

impl fmt::Display for LayerNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LayerNorm({})", self.num_features)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bn_creation() {
        let bn = BatchNormLayer::new(BatchNormConfig::new(64));
        assert_eq!(bn.gamma.len(), 64);
        assert_eq!(bn.beta.len(), 64);
        assert_eq!(bn.running_mean.len(), 64);
    }

    #[test]
    fn test_bn_param_count() {
        let bn = BatchNormLayer::new(BatchNormConfig::new(32));
        assert_eq!(bn.param_count(), 64); // 32 gamma + 32 beta
    }

    #[test]
    fn test_bn_param_count_no_affine() {
        let bn = BatchNormLayer::new(BatchNormConfig::new(32).with_no_affine());
        assert_eq!(bn.param_count(), 0);
    }

    #[test]
    fn test_bn_training_forward_normalizes() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(2));
        // 4 samples, 2 features each
        let input = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let out = bn.forward(&input, 4);
        assert_eq!(out.len(), 8);

        // Mean of normalized output per feature should be ~0
        let mean_f0 = (out[0] + out[2] + out[4] + out[6]) / 4.0;
        let mean_f1 = (out[1] + out[3] + out[5] + out[7]) / 4.0;
        assert!(mean_f0.abs() < 1e-10);
        assert!(mean_f1.abs() < 1e-10);
    }

    #[test]
    fn test_bn_inference_uses_running_stats() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(1));
        // Train on some data first
        bn.forward(&[1.0, 2.0, 3.0, 4.0], 4);
        bn.set_mode(BNMode::Inference);

        let out = bn.forward(&[2.5], 1);
        assert_eq!(out.len(), 1);
        // Should use running stats, not batch stats
    }

    #[test]
    fn test_bn_running_stats_update() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(1));
        let input = vec![0.0, 10.0];
        bn.forward(&input, 2);
        assert_eq!(bn.num_batches_tracked, 1);
        // Running mean should have been updated from 0.0 toward 5.0
        assert!(bn.running_mean[0] > 0.0);
    }

    #[test]
    fn test_bn_momentum() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(1).with_momentum(1.0));
        bn.forward(&[2.0, 4.0], 2);
        // With momentum=1.0, running mean = batch mean = 3.0
        assert!((bn.running_mean[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bn_no_affine() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(1).with_no_affine());
        let out = bn.forward(&[1.0, 3.0], 2);
        // Without affine, just normalized
        assert!((out[0] + out[1]).abs() < 1e-10); // should sum to ~0
    }

    #[test]
    fn test_bn_backward_shapes() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(3));
        bn.forward(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
        let grad_out = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let (gi, gg, gb) = bn.backward(&grad_out);
        assert_eq!(gi.len(), 6);
        assert_eq!(gg.len(), 3);
        assert_eq!(gb.len(), 3);
    }

    #[test]
    fn test_bn_fuse_params() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(2));
        bn.forward(&[1.0, 10.0, 3.0, 30.0], 2);
        bn.set_mode(BNMode::Inference);
        let (scale, offset) = bn.fuse_params();
        assert_eq!(scale.len(), 2);
        assert_eq!(offset.len(), 2);
    }

    #[test]
    fn test_bn_display() {
        let bn = BatchNormLayer::new(BatchNormConfig::new(64));
        let s = format!("{}", bn);
        assert!(s.contains("BatchNorm"));
        assert!(s.contains("64"));
        assert!(s.contains("train"));
    }

    #[test]
    fn test_bn_mode_display() {
        assert_eq!(format!("{}", BNMode::Training), "train");
        assert_eq!(format!("{}", BNMode::Inference), "eval");
    }

    #[test]
    fn test_bn_2d_forward() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(2));
        // 1 sample, 2 channels, 2x2 spatial
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = bn.forward_2d(&input, 1, 2, 2);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_layer_norm_creation() {
        let ln = LayerNorm::new(512);
        assert_eq!(ln.gamma.len(), 512);
        assert_eq!(ln.beta.len(), 512);
    }

    #[test]
    fn test_layer_norm_normalizes() {
        let ln = LayerNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = ln.forward(&input, 1);
        // Mean should be ~0
        let mean = out.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_layer_norm_batch() {
        let ln = LayerNorm::new(3);
        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let out = ln.forward(&input, 2);
        assert_eq!(out.len(), 6);
        // Each sample independently normalized
        let mean0 = (out[0] + out[1] + out[2]) / 3.0;
        let mean1 = (out[3] + out[4] + out[5]) / 3.0;
        assert!(mean0.abs() < 1e-10);
        assert!(mean1.abs() < 1e-10);
    }

    #[test]
    fn test_layer_norm_display() {
        let ln = LayerNorm::new(768);
        assert_eq!(format!("{}", ln), "LayerNorm(768)");
    }

    #[test]
    fn test_bn_constant_input() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(1));
        // All same value — variance is 0, should not crash (epsilon prevents div by zero)
        let out = bn.forward(&[5.0, 5.0, 5.0, 5.0], 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bn_single_sample() {
        let mut bn = BatchNormLayer::new(BatchNormConfig::new(2));
        let out = bn.forward(&[3.0, 7.0], 1);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
