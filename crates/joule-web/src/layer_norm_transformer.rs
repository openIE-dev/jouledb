//! Layer normalization with learnable affine parameters for transformers.
//!
//! Implements Layer Normalization (Ba et al. 2016) with configurable epsilon,
//! learnable gain (gamma) and bias (beta) parameters, and support for both
//! pre-norm and post-norm transformer architectures. Also includes RMSNorm
//! (Zhang & Sennrich 2019) as a lightweight alternative.

use std::fmt;

// ── Normalization Mode ───────────────────────────────────────────

/// Where normalization is applied relative to the sublayer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormPosition {
    /// Pre-norm: normalize before the sublayer (GPT-2, LLaMA style).
    Pre,
    /// Post-norm: normalize after the residual add (original Transformer).
    Post,
}

impl fmt::Display for NormPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pre => write!(f, "Pre-Norm"),
            Self::Post => write!(f, "Post-Norm"),
        }
    }
}

// ── Layer Normalization ──────────────────────────────────────────

/// Layer Normalization configuration.
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    pub normalized_shape: usize,
    pub eps: f64,
    pub affine: bool,
    pub position: NormPosition,
}

impl LayerNormConfig {
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
            affine: true,
            position: NormPosition::Pre,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    pub fn with_position(mut self, position: NormPosition) -> Self {
        self.position = position;
        self
    }
}

/// Layer Normalization with learnable affine parameters.
///
/// For each element in a sequence, normalizes across the feature dimension:
///   y = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// Where gamma and beta are learnable parameters of size `normalized_shape`.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub config: LayerNormConfig,
    /// Learnable scale (gamma), initialized to 1.
    pub gamma: Vec<f64>,
    /// Learnable shift (beta), initialized to 0.
    pub beta: Vec<f64>,
    /// Running statistics for diagnostics.
    running_mean_of_means: f64,
    running_mean_of_vars: f64,
    stat_count: usize,
}

impl LayerNorm {
    /// Create a new LayerNorm.
    pub fn new(config: LayerNormConfig) -> Self {
        let n = config.normalized_shape;
        Self {
            gamma: vec![1.0; n],
            beta: vec![0.0; n],
            config,
            running_mean_of_means: 0.0,
            running_mean_of_vars: 0.0,
            stat_count: 0,
        }
    }

    /// Compute mean of a slice.
    fn mean(data: &[f64]) -> f64 {
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Compute variance of a slice given the mean.
    fn variance(data: &[f64], mean: f64) -> f64 {
        data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / data.len() as f64
    }

    /// Normalize a single vector of length `normalized_shape`.
    pub fn normalize_vector(&mut self, input: &[f64]) -> Vec<f64> {
        let n = self.config.normalized_shape;
        assert_eq!(input.len(), n, "input length must match normalized_shape");

        let mu = Self::mean(input);
        let var = Self::variance(input, mu);

        // Update running stats
        self.stat_count += 1;
        let alpha = 1.0 / self.stat_count as f64;
        self.running_mean_of_means += alpha * (mu - self.running_mean_of_means);
        self.running_mean_of_vars += alpha * (var - self.running_mean_of_vars);

        let inv_std = 1.0 / (var + self.config.eps).sqrt();
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let normed = (input[i] - mu) * inv_std;
            if self.config.affine {
                output.push(self.gamma[i] * normed + self.beta[i]);
            } else {
                output.push(normed);
            }
        }
        output
    }

    /// Forward pass over a sequence: (seq_len, d) -> (seq_len, d).
    ///
    /// Each position is normalized independently across its features.
    pub fn forward(&mut self, input: &[f64], seq_len: usize) -> Vec<f64> {
        let d = self.config.normalized_shape;
        assert_eq!(input.len(), seq_len * d, "input length mismatch");
        let mut output = Vec::with_capacity(input.len());
        for s in 0..seq_len {
            let start = s * d;
            let normed = self.normalize_vector(&input[start..start + d]);
            output.extend_from_slice(&normed);
        }
        output
    }

    /// Apply normalization in the context of a residual connection.
    ///
    /// Pre-norm:  output = sublayer(norm(x)) + x
    /// Post-norm: output = norm(sublayer(x) + x)
    pub fn apply_with_residual<F>(&mut self, input: &[f64], seq_len: usize, sublayer: F) -> Vec<f64>
    where
        F: Fn(&[f64], usize) -> Vec<f64>,
    {
        let d = self.config.normalized_shape;
        match self.config.position {
            NormPosition::Pre => {
                let normed = self.forward(input, seq_len);
                let sublayer_out = sublayer(&normed, seq_len);
                // Residual add
                input.iter().zip(sublayer_out.iter())
                    .map(|(a, b)| a + b)
                    .collect()
            }
            NormPosition::Post => {
                let sublayer_out = sublayer(input, seq_len);
                // Residual add then normalize
                let residual: Vec<f64> = input.iter().zip(sublayer_out.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                self.forward(&residual, seq_len)
            }
        }
    }

    /// Retrieve running statistics for monitoring.
    pub fn running_stats(&self) -> (f64, f64, usize) {
        (self.running_mean_of_means, self.running_mean_of_vars, self.stat_count)
    }

    /// Reset running statistics.
    pub fn reset_stats(&mut self) {
        self.running_mean_of_means = 0.0;
        self.running_mean_of_vars = 0.0;
        self.stat_count = 0;
    }

    /// Total number of learnable parameters.
    pub fn num_parameters(&self) -> usize {
        if self.config.affine {
            2 * self.config.normalized_shape
        } else {
            0
        }
    }
}

impl fmt::Display for LayerNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LayerNorm(shape={}, eps={:.0e}, affine={}, pos={})",
            self.config.normalized_shape,
            self.config.eps,
            self.config.affine,
            self.config.position
        )
    }
}

// ── RMS Normalization ────────────────────────────────────────────

/// Root Mean Square Layer Normalization (RMSNorm).
///
/// Simplified variant that omits the mean centering step:
///   y = gamma * x / sqrt(mean(x^2) + eps)
///
/// Used in LLaMA, PaLM, and other efficient architectures. Slightly
/// faster than full LayerNorm due to fewer operations.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub normalized_shape: usize,
    pub eps: f64,
    /// Learnable scale parameter.
    pub gamma: Vec<f64>,
}

impl RmsNorm {
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-6,
            gamma: vec![1.0; normalized_shape],
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Compute the RMS of a slice.
    fn rms(data: &[f64]) -> f64 {
        let mean_sq = data.iter().map(|x| x * x).sum::<f64>() / data.len() as f64;
        mean_sq.sqrt()
    }

    /// Normalize a single vector.
    pub fn normalize_vector(&self, input: &[f64]) -> Vec<f64> {
        let n = self.normalized_shape;
        assert_eq!(input.len(), n, "input length must match normalized_shape");
        let rms_val = Self::rms(input);
        let inv_rms = 1.0 / (rms_val + self.eps);
        input.iter().enumerate()
            .map(|(i, &x)| self.gamma[i] * x * inv_rms)
            .collect()
    }

    /// Forward pass over a sequence.
    pub fn forward(&self, input: &[f64], seq_len: usize) -> Vec<f64> {
        let d = self.normalized_shape;
        assert_eq!(input.len(), seq_len * d, "input length mismatch");
        let mut output = Vec::with_capacity(input.len());
        for s in 0..seq_len {
            let start = s * d;
            let normed = self.normalize_vector(&input[start..start + d]);
            output.extend_from_slice(&normed);
        }
        output
    }

    /// Number of learnable parameters (just gamma).
    pub fn num_parameters(&self) -> usize {
        self.normalized_shape
    }
}

impl fmt::Display for RmsNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RMSNorm(shape={}, eps={:.0e})", self.normalized_shape, self.eps)
    }
}

// ── Batch Statistics ─────────────────────────────────────────────

/// Compute per-feature statistics across a batch of sequences.
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub feature_means: Vec<f64>,
    pub feature_vars: Vec<f64>,
    pub feature_count: usize,
}

impl FeatureStats {
    /// Compute statistics from a batch of sequences.
    ///
    /// data: (batch_size * seq_len, d_model), total_positions = batch_size * seq_len.
    pub fn compute(data: &[f64], d_model: usize, total_positions: usize) -> Self {
        assert_eq!(data.len(), total_positions * d_model, "data length mismatch");

        let mut means = vec![0.0; d_model];
        let mut vars = vec![0.0; d_model];

        // Compute means
        for pos in 0..total_positions {
            for d in 0..d_model {
                means[d] += data[pos * d_model + d];
            }
        }
        for m in means.iter_mut() {
            *m /= total_positions as f64;
        }

        // Compute variances
        for pos in 0..total_positions {
            for d in 0..d_model {
                let diff = data[pos * d_model + d] - means[d];
                vars[d] += diff * diff;
            }
        }
        for v in vars.iter_mut() {
            *v /= total_positions as f64;
        }

        Self { feature_means: means, feature_vars: vars, feature_count: d_model }
    }

    /// Max variance across features — useful for detecting exploding activations.
    pub fn max_variance(&self) -> f64 {
        self.feature_vars.iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Mean variance across features.
    pub fn mean_variance(&self) -> f64 {
        self.feature_vars.iter().sum::<f64>() / self.feature_count as f64
    }
}

impl fmt::Display for FeatureStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FeatureStats(features={}, mean_var={:.6}, max_var={:.6})",
            self.feature_count,
            self.mean_variance(),
            self.max_variance()
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let cfg = LayerNormConfig::new(4).with_affine(false);
        let mut ln = LayerNorm::new(cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.normalize_vector(&input);
        let mean: f64 = output.iter().sum::<f64>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-10));
    }

    #[test]
    fn test_layer_norm_unit_variance() {
        let cfg = LayerNormConfig::new(4).with_affine(false);
        let mut ln = LayerNorm::new(cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.normalize_vector(&input);
        let var: f64 = output.iter().map(|x| x * x).sum::<f64>() / 4.0;
        assert!(approx_eq(var, 1.0, 1e-4));
    }

    #[test]
    fn test_layer_norm_affine() {
        let cfg = LayerNormConfig::new(4);
        let mut ln = LayerNorm::new(cfg);
        // With default gamma=1, beta=0, affine should match non-affine
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let out_affine = ln.normalize_vector(&input);
        let cfg2 = LayerNormConfig::new(4).with_affine(false);
        let mut ln2 = LayerNorm::new(cfg2);
        let out_no_affine = ln2.normalize_vector(&input);
        for i in 0..4 {
            assert!(approx_eq(out_affine[i], out_no_affine[i], 1e-12));
        }
    }

    #[test]
    fn test_layer_norm_custom_gamma_beta() {
        let cfg = LayerNormConfig::new(2);
        let mut ln = LayerNorm::new(cfg);
        ln.gamma = vec![2.0, 3.0];
        ln.beta = vec![1.0, -1.0];
        let input = vec![5.0, 5.0]; // mean=5, var=0
        let output = ln.normalize_vector(&input);
        // (5-5)/sqrt(0+eps) ≈ 0, so output ≈ gamma*0 + beta = beta
        assert!(approx_eq(output[0], 1.0, 1e-2));
        assert!(approx_eq(output[1], -1.0, 1e-2));
    }

    #[test]
    fn test_layer_norm_forward_shape() {
        let cfg = LayerNormConfig::new(8);
        let mut ln = LayerNorm::new(cfg);
        let input = vec![0.5; 5 * 8]; // 5 positions x 8 features
        let output = ln.forward(&input, 5);
        assert_eq!(output.len(), 40);
    }

    #[test]
    fn test_layer_norm_running_stats() {
        let cfg = LayerNormConfig::new(4);
        let mut ln = LayerNorm::new(cfg);
        ln.normalize_vector(&[1.0, 2.0, 3.0, 4.0]);
        let (mean, var, count) = ln.running_stats();
        assert_eq!(count, 1);
        assert!(approx_eq(mean, 2.5, 1e-10));
    }

    #[test]
    fn test_layer_norm_reset_stats() {
        let cfg = LayerNormConfig::new(4);
        let mut ln = LayerNorm::new(cfg);
        ln.normalize_vector(&[1.0, 2.0, 3.0, 4.0]);
        ln.reset_stats();
        let (_, _, count) = ln.running_stats();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_layer_norm_num_params() {
        let cfg = LayerNormConfig::new(64);
        let ln = LayerNorm::new(cfg);
        assert_eq!(ln.num_parameters(), 128); // gamma + beta
    }

    #[test]
    fn test_layer_norm_no_affine_params() {
        let cfg = LayerNormConfig::new(64).with_affine(false);
        let ln = LayerNorm::new(cfg);
        assert_eq!(ln.num_parameters(), 0);
    }

    #[test]
    fn test_pre_norm_residual() {
        let cfg = LayerNormConfig::new(4).with_position(NormPosition::Pre).with_affine(false);
        let mut ln = LayerNorm::new(cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.apply_with_residual(&input, 1, |x, _| x.to_vec());
        // Pre-norm: norm(x) + x (sublayer is identity on normed input)
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_post_norm_residual() {
        let cfg = LayerNormConfig::new(4).with_position(NormPosition::Post).with_affine(false);
        let mut ln = LayerNorm::new(cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.apply_with_residual(&input, 1, |x, _| x.to_vec());
        // Post-norm: norm(x + sublayer(x)) = norm(2x), should be zero-mean
        let mean: f64 = output.iter().sum::<f64>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-10));
    }

    #[test]
    fn test_layer_norm_display() {
        let cfg = LayerNormConfig::new(256).with_position(NormPosition::Post);
        let ln = LayerNorm::new(cfg);
        let s = format!("{}", ln);
        assert!(s.contains("256"));
        assert!(s.contains("Post-Norm"));
    }

    #[test]
    fn test_rms_norm_preserves_direction() {
        let rms = RmsNorm::new(4);
        let input = vec![2.0, 0.0, 0.0, 0.0];
        let output = rms.normalize_vector(&input);
        // Only first element should be non-zero
        assert!(output[0] > 0.0);
        assert!(approx_eq(output[1], 0.0, 1e-10));
    }

    #[test]
    fn test_rms_norm_forward_shape() {
        let rms = RmsNorm::new(8);
        let input = vec![0.5; 3 * 8];
        let output = rms.forward(&input, 3);
        assert_eq!(output.len(), 24);
    }

    #[test]
    fn test_rms_norm_unit_scale() {
        let rms = RmsNorm::new(4);
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms.normalize_vector(&input);
        // RMS of [1,1,1,1] = 1, so output ≈ gamma * 1/1 = 1
        for &v in &output {
            assert!(approx_eq(v, 1.0, 0.01));
        }
    }

    #[test]
    fn test_rms_norm_display() {
        let rms = RmsNorm::new(512).with_eps(1e-8);
        let s = format!("{}", rms);
        assert!(s.contains("512"));
        assert!(s.contains("RMSNorm"));
    }

    #[test]
    fn test_feature_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 positions x 3 features
        let stats = FeatureStats::compute(&data, 3, 2);
        assert_eq!(stats.feature_count, 3);
        // Position 0: [1,2,3], Position 1: [4,5,6]
        assert!(approx_eq(stats.feature_means[0], 2.5, 1e-10)); // (1+4)/2
        assert!(approx_eq(stats.feature_means[1], 3.5, 1e-10)); // (2+5)/2
    }

    #[test]
    fn test_feature_stats_display() {
        let data = vec![1.0; 12]; // 3 x 4
        let stats = FeatureStats::compute(&data, 4, 3);
        let s = format!("{}", stats);
        assert!(s.contains("features=4"));
    }

    #[test]
    fn test_norm_position_display() {
        assert_eq!(format!("{}", NormPosition::Pre), "Pre-Norm");
        assert_eq!(format!("{}", NormPosition::Post), "Post-Norm");
    }
}
