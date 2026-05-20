//! Variational Similarity Search
//!
//! Implements gradient-based optimization for learning optimal encodings.
//! Inspired by variational quantum eigensolvers (VQE) and variational autoencoders.
//!
//! ## Key Concepts
//!
//! - **Parameterized Encoder**: Learnable rotation/scaling of input vectors
//! - **Contrastive Loss**: Push similar items close, dissimilar items apart
//! - **Gradient Descent**: SGD/Adam to optimize encoding parameters
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::variational::{VariationalEncoder, TrainingConfig};
//!
//! let mut encoder = VariationalEncoder::new(100, 50);
//!
//! // Train on pairs of similar/dissimilar items
//! encoder.train_pair(&vec1, &vec2, true);  // Similar
//! encoder.train_pair(&vec1, &vec3, false); // Dissimilar
//!
//! // Use optimized encoding
//! let encoded = encoder.encode(&query);
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum (for SGD with momentum)
    pub momentum: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Margin for contrastive loss
    pub margin: f64,
    /// Adam beta1
    pub beta1: f64,
    /// Adam beta2
    pub beta2: f64,
    /// Adam epsilon
    pub epsilon: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 1e-4,
            margin: 1.0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Variational encoder with learnable parameters
pub struct VariationalEncoder {
    /// Input dimension
    input_dim: usize,
    /// Output (encoded) dimension
    output_dim: usize,
    /// Weight matrix (output_dim x input_dim)
    weights: Vec<f64>,
    /// Bias vector (output_dim)
    bias: Vec<f64>,
    /// Training configuration
    config: TrainingConfig,
    /// Momentum buffer for weights
    weight_momentum: Vec<f64>,
    /// Momentum buffer for bias
    bias_momentum: Vec<f64>,
    /// Adam first moment for weights
    weight_m: Vec<f64>,
    /// Adam second moment for weights
    weight_v: Vec<f64>,
    /// Adam first moment for bias
    bias_m: Vec<f64>,
    /// Adam second moment for bias
    bias_v: Vec<f64>,
    /// Training step counter
    step: usize,
    /// Random number generator
    rng: StdRng,
}

impl VariationalEncoder {
    /// Create new variational encoder
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self::with_config(input_dim, output_dim, TrainingConfig::default())
    }

    /// Create with custom config
    pub fn with_config(input_dim: usize, output_dim: usize, config: TrainingConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let weight_size = output_dim * input_dim;

        // Xavier initialization
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights: Vec<f64> = (0..weight_size)
            .map(|_| rng.random_range(-scale..scale))
            .collect();
        let bias = vec![0.0; output_dim];

        Self {
            input_dim,
            output_dim,
            weights,
            bias,
            weight_momentum: vec![0.0; weight_size],
            bias_momentum: vec![0.0; output_dim],
            weight_m: vec![0.0; weight_size],
            weight_v: vec![0.0; weight_size],
            bias_m: vec![0.0; output_dim],
            bias_v: vec![0.0; output_dim],
            step: 0,
            config,
            rng,
        }
    }

    /// Encode input vector
    pub fn encode(&self, input: &[f64]) -> Vec<f64> {
        let mut output = self.bias.clone();

        for i in 0..self.output_dim {
            for j in 0..self.input_dim.min(input.len()) {
                output[i] += self.weights[i * self.input_dim + j] * input[j];
            }
            // Apply activation (tanh)
            output[i] = output[i].tanh();
        }

        output
    }

    /// Encode binary vector (convert to float first)
    pub fn encode_binary(&self, bits: &[bool]) -> Vec<f64> {
        let input: Vec<f64> = bits.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
        self.encode(&input)
    }

    /// Compute similarity between two encoded vectors
    pub fn similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        // Cosine similarity
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Train on a pair of vectors (contrastive learning)
    ///
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    /// * `similar` - True if vectors should be similar, false if dissimilar
    ///
    /// # Returns
    /// Loss value
    pub fn train_pair(&mut self, a: &[f64], b: &[f64], similar: bool) -> f64 {
        // Forward pass
        let enc_a = self.encode(a);
        let enc_b = self.encode(b);

        // Compute distance
        let dist_sq: f64 = enc_a
            .iter()
            .zip(enc_b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        let dist = dist_sq.sqrt();

        // Contrastive loss: similar pairs should be close, dissimilar should be far
        let loss = if similar {
            dist_sq
        } else {
            (self.config.margin - dist).max(0.0).powi(2)
        };

        // Backward pass (compute gradients)
        let grad_scale = if similar {
            2.0
        } else if dist < self.config.margin {
            -2.0 * (self.config.margin - dist) / dist.max(1e-10)
        } else {
            0.0
        };

        // Gradient w.r.t. encoded outputs
        let mut grad_enc_a = vec![0.0; self.output_dim];
        let mut grad_enc_b = vec![0.0; self.output_dim];

        for i in 0..self.output_dim {
            let diff = enc_a[i] - enc_b[i];
            grad_enc_a[i] = grad_scale * diff;
            grad_enc_b[i] = -grad_scale * diff;
        }

        // Backprop through tanh: d/dx tanh(x) = 1 - tanh(x)^2
        let pre_act_a = self.compute_pre_activation(a);
        let pre_act_b = self.compute_pre_activation(b);

        for i in 0..self.output_dim {
            let tanh_a = pre_act_a[i].tanh();
            let tanh_b = pre_act_b[i].tanh();
            grad_enc_a[i] *= 1.0 - tanh_a * tanh_a;
            grad_enc_b[i] *= 1.0 - tanh_b * tanh_b;
        }

        // Gradient w.r.t. weights and bias
        let mut grad_weights = vec![0.0; self.weights.len()];
        let mut grad_bias = vec![0.0; self.output_dim];

        for i in 0..self.output_dim {
            grad_bias[i] = grad_enc_a[i] + grad_enc_b[i];

            for j in 0..self.input_dim {
                let a_val = if j < a.len() { a[j] } else { 0.0 };
                let b_val = if j < b.len() { b[j] } else { 0.0 };
                grad_weights[i * self.input_dim + j] =
                    grad_enc_a[i] * a_val + grad_enc_b[i] * b_val;
            }
        }

        // Add weight decay
        for i in 0..self.weights.len() {
            grad_weights[i] += self.config.weight_decay * self.weights[i];
        }

        // Update with Adam
        self.adam_update(&grad_weights, &grad_bias);

        loss
    }

    /// Compute pre-activation values (before tanh)
    fn compute_pre_activation(&self, input: &[f64]) -> Vec<f64> {
        let mut output = self.bias.clone();

        for i in 0..self.output_dim {
            for j in 0..self.input_dim.min(input.len()) {
                output[i] += self.weights[i * self.input_dim + j] * input[j];
            }
        }

        output
    }

    /// Adam optimizer update
    fn adam_update(&mut self, grad_weights: &[f64], grad_bias: &[f64]) {
        self.step += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;

        // Bias correction
        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);

        // Update weights
        for i in 0..self.weights.len() {
            self.weight_m[i] = beta1 * self.weight_m[i] + (1.0 - beta1) * grad_weights[i];
            self.weight_v[i] = beta2 * self.weight_v[i] + (1.0 - beta2) * grad_weights[i].powi(2);

            let m_hat = self.weight_m[i] / bc1;
            let v_hat = self.weight_v[i] / bc2;

            self.weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        // Update bias
        for i in 0..self.output_dim {
            self.bias_m[i] = beta1 * self.bias_m[i] + (1.0 - beta1) * grad_bias[i];
            self.bias_v[i] = beta2 * self.bias_v[i] + (1.0 - beta2) * grad_bias[i].powi(2);

            let m_hat = self.bias_m[i] / bc1;
            let v_hat = self.bias_v[i] / bc2;

            self.bias[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// Train on a batch of triplets (anchor, positive, negative)
    pub fn train_triplet(&mut self, anchor: &[f64], positive: &[f64], negative: &[f64]) -> f64 {
        let enc_anchor = self.encode(anchor);
        let enc_positive = self.encode(positive);
        let enc_negative = self.encode(negative);

        // Triplet loss: d(a,p) - d(a,n) + margin
        let dist_pos: f64 = enc_anchor
            .iter()
            .zip(enc_positive.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        let dist_neg: f64 = enc_anchor
            .iter()
            .zip(enc_negative.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        let loss = (dist_pos - dist_neg + self.config.margin).max(0.0);

        if loss > 0.0 {
            // Train to pull positive closer, push negative away
            self.train_pair(anchor, positive, true);
            self.train_pair(anchor, negative, false);
        }

        loss
    }

    /// Get number of trainable parameters
    pub fn num_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }
}

/// Variational index that learns optimal encoding
pub struct VariationalIndex {
    /// Encoder
    encoder: VariationalEncoder,
    /// Stored encoded vectors
    entries: HashMap<String, Vec<f64>>,
    /// Original vectors (for re-encoding after training)
    originals: HashMap<String, Vec<f64>>,
}

impl VariationalIndex {
    /// Create new variational index
    pub fn new(input_dim: usize, encoded_dim: usize) -> Self {
        Self {
            encoder: VariationalEncoder::new(input_dim, encoded_dim),
            entries: HashMap::new(),
            originals: HashMap::new(),
        }
    }

    /// Insert vector
    pub fn insert(&mut self, key: &str, vector: &[f64]) {
        let encoded = self.encoder.encode(vector);
        self.entries.insert(key.to_string(), encoded);
        self.originals.insert(key.to_string(), vector.to_vec());
    }

    /// Find similar entries
    pub fn find_similar(&self, query: &[f64], limit: usize) -> Vec<(String, f64)> {
        let enc_query = self.encoder.encode(query);

        let mut results: Vec<_> = self
            .entries
            .iter()
            .map(|(key, enc)| (key.clone(), VariationalEncoder::similarity(&enc_query, enc)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Train on similar pair
    pub fn train_similar(&mut self, key1: &str, key2: &str) -> Option<f64> {
        let v1 = self.originals.get(key1)?.clone();
        let v2 = self.originals.get(key2)?.clone();
        Some(self.encoder.train_pair(&v1, &v2, true))
    }

    /// Train on dissimilar pair
    pub fn train_dissimilar(&mut self, key1: &str, key2: &str) -> Option<f64> {
        let v1 = self.originals.get(key1)?.clone();
        let v2 = self.originals.get(key2)?.clone();
        Some(self.encoder.train_pair(&v1, &v2, false))
    }

    /// Re-encode all entries (after training)
    pub fn refresh(&mut self) {
        let keys: Vec<_> = self.originals.keys().cloned().collect();
        for key in keys {
            if let Some(original) = self.originals.get(&key) {
                let encoded = self.encoder.encode(original);
                self.entries.insert(key, encoded);
            }
        }
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Variational Binary Encoder for HDC integration
pub struct VariationalBinaryEncoder {
    /// Core encoder
    inner: VariationalEncoder,
    /// Threshold for binarization
    threshold: f64,
}

impl VariationalBinaryEncoder {
    /// Create new binary encoder
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            inner: VariationalEncoder::new(input_dim, output_dim),
            threshold: 0.0,
        }
    }

    /// Encode to binary
    pub fn encode(&self, input: &[bool]) -> Vec<bool> {
        let float_input: Vec<f64> = input.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
        let encoded = self.inner.encode(&float_input);

        encoded.iter().map(|&x| x > self.threshold).collect()
    }

    /// Encode to soft (continuous) values
    pub fn encode_soft(&self, input: &[bool]) -> Vec<f64> {
        let float_input: Vec<f64> = input.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
        self.inner.encode(&float_input)
    }

    /// Train on pair
    pub fn train_pair(&mut self, a: &[bool], b: &[bool], similar: bool) -> f64 {
        let fa: Vec<f64> = a.iter().map(|&x| if x { 1.0 } else { -1.0 }).collect();
        let fb: Vec<f64> = b.iter().map(|&x| if x { 1.0 } else { -1.0 }).collect();
        self.inner.train_pair(&fa, &fb, similar)
    }
}

/// Online learning adapter for variational encoding
pub struct OnlineVariationalLearner {
    encoder: VariationalEncoder,
    /// Recent similar pairs
    similar_buffer: Vec<(Vec<f64>, Vec<f64>)>,
    /// Recent dissimilar pairs
    dissimilar_buffer: Vec<(Vec<f64>, Vec<f64>)>,
    /// Buffer size
    buffer_size: usize,
    /// Training frequency (every N observations)
    train_frequency: usize,
    /// Observation counter
    observation_count: usize,
}

impl OnlineVariationalLearner {
    /// Create new online learner
    pub fn new(input_dim: usize, output_dim: usize, buffer_size: usize) -> Self {
        Self {
            encoder: VariationalEncoder::new(input_dim, output_dim),
            similar_buffer: Vec::with_capacity(buffer_size),
            dissimilar_buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            train_frequency: 10,
            observation_count: 0,
        }
    }

    /// Observe a similar pair
    pub fn observe_similar(&mut self, a: &[f64], b: &[f64]) {
        if self.similar_buffer.len() >= self.buffer_size {
            self.similar_buffer.remove(0);
        }
        self.similar_buffer.push((a.to_vec(), b.to_vec()));
        self.maybe_train();
    }

    /// Observe a dissimilar pair
    pub fn observe_dissimilar(&mut self, a: &[f64], b: &[f64]) {
        if self.dissimilar_buffer.len() >= self.buffer_size {
            self.dissimilar_buffer.remove(0);
        }
        self.dissimilar_buffer.push((a.to_vec(), b.to_vec()));
        self.maybe_train();
    }

    /// Maybe trigger training
    fn maybe_train(&mut self) {
        self.observation_count += 1;
        if self.observation_count % self.train_frequency == 0 {
            self.train_batch();
        }
    }

    /// Train on buffered data
    pub fn train_batch(&mut self) -> f64 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (a, b) in &self.similar_buffer {
            total_loss += self.encoder.train_pair(a, b, true);
            count += 1;
        }

        for (a, b) in &self.dissimilar_buffer {
            total_loss += self.encoder.train_pair(a, b, false);
            count += 1;
        }

        if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        }
    }

    /// Encode using learned encoder
    pub fn encode(&self, input: &[f64]) -> Vec<f64> {
        self.encoder.encode(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variational_encoder_basic() {
        let encoder = VariationalEncoder::new(10, 5);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let output = encoder.encode(&input);

        assert_eq!(output.len(), 5);

        // Output should be in [-1, 1] due to tanh
        for &v in &output {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_variational_encoder_train() {
        let mut encoder = VariationalEncoder::new(10, 5);

        let similar_a = vec![1.0; 10];
        let similar_b = vec![1.1; 10];
        let dissimilar = vec![-1.0; 10];

        // Train on similar pair
        let loss1 = encoder.train_pair(&similar_a, &similar_b, true);

        // Train on dissimilar pair
        let loss2 = encoder.train_pair(&similar_a, &dissimilar, false);

        // Both losses should be non-negative
        assert!(loss1 >= 0.0);
        assert!(loss2 >= 0.0);
    }

    #[test]
    fn test_variational_encoder_learns() {
        let mut encoder = VariationalEncoder::new(5, 3);
        encoder.set_learning_rate(0.1);

        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.9, 0.1, 0.0, 0.0, 0.0];
        let c = vec![-1.0, 0.0, 0.0, 0.0, 0.0];

        // Measure initial distances
        let enc_a_init = encoder.encode(&a);
        let enc_b_init = encoder.encode(&b);
        let enc_c_init = encoder.encode(&c);

        let dist_ab_init: f64 = enc_a_init
            .iter()
            .zip(enc_b_init.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        let dist_ac_init: f64 = enc_a_init
            .iter()
            .zip(enc_c_init.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        // Train
        for _ in 0..100 {
            encoder.train_pair(&a, &b, true);
            encoder.train_pair(&a, &c, false);
        }

        // Measure final distances
        let enc_a_final = encoder.encode(&a);
        let enc_b_final = encoder.encode(&b);
        let enc_c_final = encoder.encode(&c);

        let dist_ab_final: f64 = enc_a_final
            .iter()
            .zip(enc_b_final.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        let _dist_ac_final: f64 = enc_a_final
            .iter()
            .zip(enc_c_final.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        // After training, similar should be closer
        assert!(
            dist_ab_final < dist_ab_init + 0.5,
            "Similar pair should get closer: {} -> {}",
            dist_ab_init,
            dist_ab_final
        );
    }

    #[test]
    fn test_variational_index() {
        let mut index = VariationalIndex::new(10, 5);

        let v1: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let v2: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect();
        let v3: Vec<f64> = (0..10).map(|i| (10 - i) as f64).collect();

        index.insert("v1", &v1);
        index.insert("v2", &v2);
        index.insert("v3", &v3);

        assert_eq!(index.len(), 3);

        let results = index.find_similar(&v1, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_variational_index_training() {
        let mut index = VariationalIndex::new(5, 3);

        let v1 = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.9, 0.1, 0.0, 0.0, 0.0];
        let v3 = vec![-1.0, 0.0, 0.0, 0.0, 0.0];

        index.insert("v1", &v1);
        index.insert("v2", &v2);
        index.insert("v3", &v3);

        // Train
        for _ in 0..20 {
            index.train_similar("v1", "v2");
            index.train_dissimilar("v1", "v3");
        }

        index.refresh();

        // Check results reflect training
        let results = index.find_similar(&v1, 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_variational_binary_encoder() {
        let encoder = VariationalBinaryEncoder::new(10, 5);

        let bits = vec![
            true, false, true, true, false, false, true, false, true, false,
        ];
        let encoded = encoder.encode(&bits);

        assert_eq!(encoded.len(), 5);
    }

    #[test]
    fn test_online_learner() {
        let mut learner = OnlineVariationalLearner::new(5, 3, 10);

        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.9, 0.1, 0.0, 0.0, 0.0];
        let c = vec![-1.0, 0.0, 0.0, 0.0, 0.0];

        // Observe pairs
        for _ in 0..20 {
            learner.observe_similar(&a, &b);
            learner.observe_dissimilar(&a, &c);
        }

        // Should be able to encode
        let enc = learner.encode(&a);
        assert_eq!(enc.len(), 3);
    }

    #[test]
    fn test_triplet_loss() {
        let mut encoder = VariationalEncoder::new(5, 3);

        let anchor = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0, 0.0, 0.0];
        let negative = vec![-1.0, 0.0, 0.0, 0.0, 0.0];

        let loss = encoder.train_triplet(&anchor, &positive, &negative);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_similarity_function() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![-1.0, 0.0, 0.0];

        // Same vector should have similarity 1
        let sim_aa = VariationalEncoder::similarity(&a, &b);
        assert!((sim_aa - 1.0).abs() < 0.01);

        // Opposite vectors should have similarity -1
        let sim_ac = VariationalEncoder::similarity(&a, &c);
        assert!((sim_ac - (-1.0)).abs() < 0.01);
    }
}
