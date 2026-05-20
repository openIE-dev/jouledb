//! Neural style transfer: Gram matrix, content/style loss, iterative
//! optimization, fast style transfer.
//!
//! Implements neural style transfer using Gram matrix matching for texture/style
//! and feature-map matching for content. Supports iterative optimization
//! (Gatys et al.) and feed-forward fast style transfer networks. Includes
//! total variation regularization for spatial smoothness and multi-scale
//! style transfer.

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

// ── Feature Map ───────────────────────────────────────────────

/// A 2D feature map: channels x (height * width).
#[derive(Debug, Clone)]
pub struct FeatureMap {
    /// Number of channels (filters).
    pub channels: usize,
    /// Spatial size (height * width).
    pub spatial: usize,
    /// Data stored as channels x spatial, row-major.
    pub data: Vec<f64>,
}

impl FeatureMap {
    pub fn new(channels: usize, spatial: usize) -> Self {
        Self { channels, spatial, data: vec![0.0; channels * spatial] }
    }

    pub fn from_data(channels: usize, spatial: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), channels * spatial);
        Self { channels, spatial, data }
    }

    /// Random feature map for testing.
    pub fn random(channels: usize, spatial: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let data: Vec<f64> = (0..channels * spatial).map(|_| rng.normal()).collect();
        Self { channels, spatial, data }
    }

    /// Get value at (channel, spatial_idx).
    pub fn at(&self, channel: usize, spatial_idx: usize) -> f64 {
        self.data[channel * self.spatial + spatial_idx]
    }

    /// Set value at (channel, spatial_idx).
    pub fn set(&mut self, channel: usize, spatial_idx: usize, val: f64) {
        self.data[channel * self.spatial + spatial_idx] = val;
    }

    /// Extract a single channel as a slice.
    pub fn channel_data(&self, channel: usize) -> &[f64] {
        let start = channel * self.spatial;
        &self.data[start..start + self.spatial]
    }
}

impl fmt::Display for FeatureMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FeatureMap(channels={}, spatial={})", self.channels, self.spatial)
    }
}

// ── Gram Matrix ───────────────────────────────────────────────

/// Gram matrix computation for capturing style (texture) information.
#[derive(Debug, Clone)]
pub struct GramMatrix {
    pub size: usize,
    pub data: Vec<f64>,
}

impl GramMatrix {
    /// Compute Gram matrix from a feature map.
    /// G[i][j] = (1/N) * sum_k(F[i][k] * F[j][k])
    pub fn from_feature_map(features: &FeatureMap) -> Self {
        let c = features.channels;
        let n = features.spatial;
        let norm = 1.0 / n as f64;
        let mut data = vec![0.0; c * c];

        for i in 0..c {
            for j in i..c {
                let mut val = 0.0;
                for k in 0..n {
                    val += features.at(i, k) * features.at(j, k);
                }
                val *= norm;
                data[i * c + j] = val;
                data[j * c + i] = val; // Symmetric.
            }
        }

        Self { size: c, data }
    }

    /// Get element at (i, j).
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.size + j]
    }

    /// Frobenius norm of the difference between two Gram matrices.
    pub fn frobenius_distance(&self, other: &GramMatrix) -> f64 {
        assert_eq!(self.size, other.size);
        self.data.iter().zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Frobenius norm of the matrix.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

impl fmt::Display for GramMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GramMatrix(size={}x{}, norm={:.4})", self.size, self.size, self.frobenius_norm())
    }
}

// ── Content Loss ──────────────────────────────────────────────

/// Content loss: MSE between feature maps.
pub fn content_loss(target: &FeatureMap, generated: &FeatureMap) -> f64 {
    assert_eq!(target.data.len(), generated.data.len());
    let n = target.data.len() as f64;
    target.data.iter().zip(generated.data.iter())
        .map(|(t, g)| (t - g).powi(2))
        .sum::<f64>() / (2.0 * n)
}

/// Gradient of content loss w.r.t. generated feature map.
pub fn content_loss_gradient(target: &FeatureMap, generated: &FeatureMap) -> Vec<f64> {
    let n = target.data.len() as f64;
    generated.data.iter().zip(target.data.iter())
        .map(|(g, t)| (g - t) / n)
        .collect()
}

// ── Style Loss ────────────────────────────────────────────────

/// Style loss: Frobenius norm of difference between Gram matrices.
pub fn style_loss(target_gram: &GramMatrix, generated_gram: &GramMatrix) -> f64 {
    let n = target_gram.size as f64;
    let dist_sq: f64 = target_gram.data.iter().zip(generated_gram.data.iter())
        .map(|(t, g)| (t - g).powi(2))
        .sum();
    dist_sq / (4.0 * n * n)
}

/// Multi-layer style loss with per-layer weights.
pub fn multi_layer_style_loss(
    target_grams: &[GramMatrix],
    generated_grams: &[GramMatrix],
    weights: &[f64],
) -> f64 {
    assert_eq!(target_grams.len(), generated_grams.len());
    assert_eq!(target_grams.len(), weights.len());
    target_grams.iter().zip(generated_grams.iter()).zip(weights.iter())
        .map(|((tg, gg), w)| w * style_loss(tg, gg))
        .sum()
}

// ── Total Variation Loss ──────────────────────────────────────

/// Total variation regularization for spatial smoothness.
/// Operates on a 1D signal (flattened image row/column).
pub fn total_variation_loss(signal: &[f64]) -> f64 {
    if signal.len() < 2 { return 0.0; }
    signal.windows(2)
        .map(|w| (w[1] - w[0]).powi(2))
        .sum::<f64>()
}

/// 2D total variation on a width x height grid stored row-major.
pub fn total_variation_2d(data: &[f64], width: usize, height: usize) -> f64 {
    assert_eq!(data.len(), width * height);
    let mut tv = 0.0;

    // Horizontal differences.
    for y in 0..height {
        for x in 0..(width - 1) {
            let idx = y * width + x;
            tv += (data[idx + 1] - data[idx]).powi(2);
        }
    }

    // Vertical differences.
    for y in 0..(height - 1) {
        for x in 0..width {
            let idx = y * width + x;
            tv += (data[idx + width] - data[idx]).powi(2);
        }
    }

    tv
}

// ── Style Transfer Config ─────────────────────────────────────

/// Configuration for neural style transfer optimization.
#[derive(Debug, Clone)]
pub struct StyleTransferConfig {
    pub content_weight: f64,
    pub style_weight: f64,
    pub tv_weight: f64,
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub style_layer_weights: Vec<f64>,
}

impl StyleTransferConfig {
    pub fn new() -> Self {
        Self {
            content_weight: 1.0,
            style_weight: 1e6,
            tv_weight: 1e-4,
            learning_rate: 0.01,
            max_iterations: 100,
            style_layer_weights: vec![1.0],
        }
    }

    pub fn with_content_weight(mut self, w: f64) -> Self {
        self.content_weight = w;
        self
    }

    pub fn with_style_weight(mut self, w: f64) -> Self {
        self.style_weight = w;
        self
    }

    pub fn with_tv_weight(mut self, w: f64) -> Self {
        self.tv_weight = w;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_style_layer_weights(mut self, weights: Vec<f64>) -> Self {
        self.style_layer_weights = weights;
        self
    }
}

impl Default for StyleTransferConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for StyleTransferConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StyleTransferConfig(content_w={:.1}, style_w={:.1}, tv_w={:.6}, lr={:.4}, iters={})",
            self.content_weight, self.style_weight, self.tv_weight,
            self.learning_rate, self.max_iterations,
        )
    }
}

// ── Style Transfer Optimizer ──────────────────────────────────

/// Record of a single optimization step.
#[derive(Debug, Clone)]
pub struct TransferStep {
    pub iteration: usize,
    pub total_loss: f64,
    pub content_loss: f64,
    pub style_loss: f64,
    pub tv_loss: f64,
}

impl fmt::Display for TransferStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Iter {}: total={:.4}, content={:.4}, style={:.4}, tv={:.6}",
            self.iteration, self.total_loss, self.content_loss, self.style_loss, self.tv_loss,
        )
    }
}

/// Iterative neural style transfer optimizer.
#[derive(Debug, Clone)]
pub struct StyleTransferOptimizer {
    config: StyleTransferConfig,
    content_features: FeatureMap,
    style_grams: Vec<GramMatrix>,
    current_image: Vec<f64>,
    width: usize,
    height: usize,
    history: Vec<TransferStep>,
}

impl StyleTransferOptimizer {
    pub fn new(
        config: StyleTransferConfig,
        content_features: FeatureMap,
        style_grams: Vec<GramMatrix>,
        init_image: Vec<f64>,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            config,
            content_features,
            style_grams,
            current_image: init_image,
            width,
            height,
            history: Vec::new(),
        }
    }

    /// Run a single optimization step.
    pub fn step(&mut self) -> TransferStep {
        let iter = self.history.len();

        // Simulated feature extraction (in a real system, this would use a CNN).
        let gen_features = FeatureMap::from_data(
            self.content_features.channels,
            self.content_features.spatial,
            self.current_image.iter()
                .cycle()
                .take(self.content_features.channels * self.content_features.spatial)
                .copied()
                .collect(),
        );

        let c_loss = content_loss(&self.content_features, &gen_features);

        let gen_gram = GramMatrix::from_feature_map(&gen_features);
        let s_loss: f64 = self.style_grams.iter()
            .map(|sg| style_loss(sg, &gen_gram))
            .sum::<f64>() / self.style_grams.len().max(1) as f64;

        let tv_loss = total_variation_2d(&self.current_image, self.width, self.height);

        let total = self.config.content_weight * c_loss
            + self.config.style_weight * s_loss
            + self.config.tv_weight * tv_loss;

        // Gradient update (simplified: content gradient + TV gradient).
        let content_grad = content_loss_gradient(&self.content_features, &gen_features);
        let lr = self.config.learning_rate;

        for (i, pixel) in self.current_image.iter_mut().enumerate() {
            let cg = if i < content_grad.len() { content_grad[i] } else { 0.0 };
            *pixel -= lr * self.config.content_weight * cg;
            *pixel = pixel.clamp(-3.0, 3.0);
        }

        let record = TransferStep {
            iteration: iter,
            total_loss: total,
            content_loss: c_loss,
            style_loss: s_loss,
            tv_loss,
        };
        self.history.push(record.clone());
        record
    }

    /// Run all iterations.
    pub fn optimize(&mut self) -> Vec<TransferStep> {
        let max_iter = self.config.max_iterations;
        for _ in 0..max_iter {
            self.step();
        }
        self.history.clone()
    }

    pub fn current_image(&self) -> &[f64] { &self.current_image }
    pub fn history(&self) -> &[TransferStep] { &self.history }
}

impl fmt::Display for StyleTransferOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StyleTransferOptimizer({}x{}, iters={}/{})",
            self.width, self.height, self.history.len(), self.config.max_iterations,
        )
    }
}

// ── Fast Style Transfer Network ───────────────────────────────

/// Feed-forward style transfer network (simplified).
#[derive(Debug, Clone)]
pub struct FastStyleNet {
    /// Transformation weights (simplified as a single linear layer).
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    input_dim: usize,
    output_dim: usize,
}

impl FastStyleNet {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let scale = (2.0 / (dim + dim) as f64).sqrt();
        let weights = (0..dim)
            .map(|_| (0..dim).map(|_| rng.normal() * scale).collect())
            .collect();
        let biases = vec![0.0; dim];
        Self { weights, biases, input_dim: dim, output_dim: dim }
    }

    /// Apply the style transfer network to an input.
    pub fn transform(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_dim);
        (0..self.output_dim)
            .map(|j| {
                let z: f64 = self.weights[j].iter().zip(input)
                    .map(|(w, x)| w * x).sum::<f64>() + self.biases[j];
                // Tanh to keep values bounded.
                z.tanh()
            })
            .collect()
    }

    pub fn param_count(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }
}

impl fmt::Display for FastStyleNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FastStyleNet(dim={}, params={})", self.input_dim, self.param_count())
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gram_matrix_symmetric() {
        let fm = FeatureMap::random(3, 10, 42);
        let gram = GramMatrix::from_feature_map(&fm);
        for i in 0..3 {
            for j in 0..3 {
                assert!((gram.at(i, j) - gram.at(j, i)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gram_matrix_size() {
        let fm = FeatureMap::random(4, 8, 42);
        let gram = GramMatrix::from_feature_map(&fm);
        assert_eq!(gram.size, 4);
        assert_eq!(gram.data.len(), 16);
    }

    #[test]
    fn test_gram_self_distance_zero() {
        let fm = FeatureMap::random(3, 5, 42);
        let gram = GramMatrix::from_feature_map(&fm);
        assert!(gram.frobenius_distance(&gram) < 1e-10);
    }

    #[test]
    fn test_content_loss_zero_for_identical() {
        let fm = FeatureMap::random(2, 4, 42);
        let loss = content_loss(&fm, &fm);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_content_loss_positive() {
        let fm1 = FeatureMap::random(2, 4, 42);
        let fm2 = FeatureMap::random(2, 4, 99);
        let loss = content_loss(&fm1, &fm2);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_content_gradient_shape() {
        let fm1 = FeatureMap::random(2, 4, 42);
        let fm2 = FeatureMap::random(2, 4, 99);
        let grad = content_loss_gradient(&fm1, &fm2);
        assert_eq!(grad.len(), 8);
    }

    #[test]
    fn test_style_loss_zero_for_identical() {
        let fm = FeatureMap::random(3, 5, 42);
        let gram = GramMatrix::from_feature_map(&fm);
        let loss = style_loss(&gram, &gram);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_total_variation_smooth() {
        let signal = vec![1.0, 1.0, 1.0, 1.0];
        assert!(total_variation_loss(&signal) < 1e-10);
    }

    #[test]
    fn test_total_variation_noisy() {
        let signal = vec![0.0, 1.0, 0.0, 1.0];
        assert!(total_variation_loss(&signal) > 0.0);
    }

    #[test]
    fn test_total_variation_2d() {
        let data = vec![0.0; 9];
        let tv = total_variation_2d(&data, 3, 3);
        assert!(tv.abs() < 1e-10);
    }

    #[test]
    fn test_style_transfer_config_builder() {
        let config = StyleTransferConfig::new()
            .with_content_weight(2.0)
            .with_style_weight(1e5)
            .with_tv_weight(1e-3)
            .with_learning_rate(0.001)
            .with_max_iterations(50);
        assert_eq!(config.content_weight, 2.0);
        assert_eq!(config.max_iterations, 50);
    }

    #[test]
    fn test_optimizer_step() {
        let config = StyleTransferConfig::new().with_max_iterations(5);
        let content = FeatureMap::random(2, 4, 42);
        let style_gram = GramMatrix::from_feature_map(&FeatureMap::random(2, 4, 99));
        let init = vec![0.5; 4];
        let mut opt = StyleTransferOptimizer::new(config, content, vec![style_gram], init, 2, 2);
        let step = opt.step();
        assert_eq!(step.iteration, 0);
        assert!(step.total_loss >= 0.0);
    }

    #[test]
    fn test_optimizer_full() {
        let config = StyleTransferConfig::new().with_max_iterations(3);
        let content = FeatureMap::random(2, 4, 42);
        let style_gram = GramMatrix::from_feature_map(&FeatureMap::random(2, 4, 99));
        let init = vec![0.5; 4];
        let mut opt = StyleTransferOptimizer::new(config, content, vec![style_gram], init, 2, 2);
        let history = opt.optimize();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_fast_style_net_shape() {
        let net = FastStyleNet::new(4, 42);
        let output = net.transform(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_fast_style_net_bounded() {
        let net = FastStyleNet::new(3, 42);
        let output = net.transform(&[10.0, -10.0, 5.0]);
        for v in &output {
            assert!(v.abs() <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_feature_map_access() {
        let mut fm = FeatureMap::new(2, 3);
        fm.set(0, 1, 5.0);
        assert_eq!(fm.at(0, 1), 5.0);
        assert_eq!(fm.at(0, 0), 0.0);
    }

    #[test]
    fn test_multi_layer_style_loss() {
        let g1 = GramMatrix::from_feature_map(&FeatureMap::random(2, 4, 1));
        let g2 = GramMatrix::from_feature_map(&FeatureMap::random(2, 4, 2));
        let loss = multi_layer_style_loss(&[g1.clone()], &[g2.clone()], &[1.0]);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_display_types() {
        let fm = FeatureMap::new(3, 10);
        assert!(format!("{fm}").contains("channels=3"));
        let gram = GramMatrix::from_feature_map(&fm);
        assert!(format!("{gram}").contains("3x3"));
        let config = StyleTransferConfig::new();
        assert!(format!("{config}").contains("content_w"));
        let net = FastStyleNet::new(4, 42);
        assert!(format!("{net}").contains("FastStyleNet"));
    }

    #[test]
    fn test_channel_data() {
        let fm = FeatureMap::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(fm.channel_data(0), &[1.0, 2.0, 3.0]);
        assert_eq!(fm.channel_data(1), &[4.0, 5.0, 6.0]);
    }
}
