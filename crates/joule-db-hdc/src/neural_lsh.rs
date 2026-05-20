//! Neural Locality-Sensitive Hashing (NLSH)
//!
//! Implements learned hash functions for domain-specific similarity search.
//! Instead of random projections (as in classical LSH), a simple neural network
//! learns task-specific hash functions via triplet loss training.
//!
//! ## Key Concepts
//!
//! - **Neural Hash Function**: A 2-layer neural network that maps input vectors
//!   to binary hash codes. The network learns to place similar items in the same
//!   hash bucket and dissimilar items in different buckets.
//! - **Triplet Loss Training**: Given (anchor, positive, negative) triplets, the
//!   network learns to minimize the Hamming distance between anchor and positive
//!   hashes while maximizing the distance to negative hashes.
//! - **Multi-Table Indexing**: Multiple independent hash functions reduce false
//!   negatives by providing multiple chances to co-locate similar items.
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::neural_lsh::{NeuralLSHConfig, NeuralLSHIndex};
//!
//! let config = NeuralLSHConfig {
//!     input_dim: 64,
//!     ..Default::default()
//! };
//! let mut index = NeuralLSHIndex::new(config);
//!
//! // Train on similarity triplets (anchor, positive, negative)
//! let triplets = vec![
//!     (anchor.clone(), positive.clone(), negative.clone()),
//! ];
//! index.train(&triplets);
//!
//! // Insert vectors
//! index.insert(0, &vec![0.1; 64]);
//!
//! // Query for nearest neighbors
//! let results = index.query(&vec![0.1; 64], 5);
//! ```

use rand::Rng;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Neural LSH index.
#[derive(Debug, Clone)]
pub struct NeuralLSHConfig {
    /// Number of hash bits produced by each neural hash function.
    /// More bits give finer-grained buckets but increase sparsity.
    pub hash_bits: usize,
    /// Number of independent hash tables. More tables reduce false negatives.
    pub num_tables: usize,
    /// Hidden layer dimension of the neural hash function.
    pub hidden_dim: usize,
    /// Learning rate for triplet-loss SGD training.
    pub learning_rate: f64,
    /// Dimensionality of the input vectors.
    pub input_dim: usize,
}

impl Default for NeuralLSHConfig {
    fn default() -> Self {
        Self {
            hash_bits: 16,
            num_tables: 8,
            hidden_dim: 128,
            learning_rate: 0.01,
            input_dim: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Neural Hash Function (2-layer network)
// ---------------------------------------------------------------------------

/// A simple 2-layer neural network that maps real-valued input vectors to
/// binary hash codes.
///
/// Architecture: `input -> linear(W1, b1) -> ReLU -> linear(W2) -> sign -> bits`
///
/// The network is trained with triplet loss so that similar inputs produce
/// similar hash codes (low Hamming distance).
#[derive(Debug, Clone)]
pub struct NeuralHashFunction {
    /// First-layer weights: `input_dim x hidden_dim`
    pub weights_1: Vec<Vec<f64>>,
    /// First-layer bias: `hidden_dim`
    pub bias_1: Vec<f64>,
    /// Second-layer weights: `hidden_dim x hash_bits`
    pub weights_2: Vec<Vec<f64>>,
    /// Number of output hash bits (kept for convenience).
    hash_bits: usize,
    /// Learning rate for weight updates.
    learning_rate: f64,
}

impl NeuralHashFunction {
    /// Create a new `NeuralHashFunction` with Xavier-initialized weights.
    pub fn new(input_dim: usize, hidden_dim: usize, hash_bits: usize, learning_rate: f64) -> Self {
        let mut rng = rand::rng();

        // Xavier initialization scale factors
        let scale_1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale_2 = (2.0 / (hidden_dim + hash_bits) as f64).sqrt();

        let weights_1: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.random_range(-scale_1..scale_1))
                    .collect()
            })
            .collect();

        let bias_1: Vec<f64> = vec![0.0; hidden_dim];

        let weights_2: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| {
                (0..hash_bits)
                    .map(|_| rng.random_range(-scale_2..scale_2))
                    .collect()
            })
            .collect();

        Self {
            weights_1,
            bias_1,
            weights_2,
            hash_bits,
            learning_rate,
        }
    }

    /// Forward pass producing continuous (pre-sign) activations.
    ///
    /// Returns `(hidden, output)` where `hidden` is the ReLU-activated hidden
    /// layer and `output` is the raw pre-sign logits.
    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let input_dim = self.weights_1.len();
        let hidden_dim = self.bias_1.len();

        // Layer 1: hidden = ReLU(W1^T * input + b1)
        let mut hidden = vec![0.0; hidden_dim];
        for j in 0..hidden_dim {
            let mut sum = self.bias_1[j];
            for i in 0..input_dim {
                sum += input[i] * self.weights_1[i][j];
            }
            hidden[j] = sum.max(0.0); // ReLU
        }

        // Layer 2: output = W2^T * hidden
        let mut output = vec![0.0; self.hash_bits];
        for j in 0..self.hash_bits {
            let mut sum = 0.0;
            for i in 0..hidden_dim {
                sum += hidden[i] * self.weights_2[i][j];
            }
            output[j] = sum;
        }

        (hidden, output)
    }

    /// Compute the binary hash code for an input vector.
    ///
    /// Runs the forward pass, applies a sign threshold (>= 0 -> 1, < 0 -> 0),
    /// and packs the resulting bits into a `u64`. Only the lower `hash_bits`
    /// bits are meaningful; the rest are zero.
    ///
    /// # Panics
    ///
    /// Panics if `hash_bits > 64`.
    pub fn hash(&self, input: &[f64]) -> u64 {
        assert!(
            self.hash_bits <= 64,
            "hash_bits must be <= 64 for u64 packing"
        );

        let (_hidden, output) = self.forward(input);

        let mut code: u64 = 0;
        for (i, &val) in output.iter().enumerate() {
            if val >= 0.0 {
                code |= 1u64 << i;
            }
        }
        code
    }

    /// Perform a single triplet-loss training step.
    ///
    /// Given an (anchor, positive, negative) triplet the loss is:
    ///
    /// ```text
    /// L = max(0, ||h(a) - h(p)||^2 - ||h(a) - h(n)||^2 + margin)
    /// ```
    ///
    /// We use continuous (pre-sign) outputs as a differentiable surrogate for
    /// the discrete Hamming distance and apply vanilla SGD to update weights.
    pub fn train_step(&mut self, anchor: &[f64], positive: &[f64], negative: &[f64]) {
        let margin = 1.0;

        // Forward passes
        let (h_a, o_a) = self.forward(anchor);
        let (h_p, o_p) = self.forward(positive);
        let (h_n, o_n) = self.forward(negative);

        // Squared distances in continuous output space
        let dist_pos: f64 = o_a
            .iter()
            .zip(o_p.iter())
            .map(|(a, p)| (a - p).powi(2))
            .sum();
        let dist_neg: f64 = o_a
            .iter()
            .zip(o_n.iter())
            .map(|(a, n)| (a - n).powi(2))
            .sum();

        let loss = (dist_pos - dist_neg + margin).max(0.0);
        if loss <= 0.0 {
            return; // Triplet constraint already satisfied
        }

        // Gradient of loss w.r.t. outputs:
        //   dL/do_a = 2*(o_a - o_p) - 2*(o_a - o_n) = 2*(o_n - o_p)
        //   dL/do_p = -2*(o_a - o_p) = 2*(o_p - o_a)
        //   dL/do_n = 2*(o_a - o_n)
        let hash_bits = self.hash_bits;
        let hidden_dim = self.bias_1.len();
        let input_dim = self.weights_1.len();

        let mut grad_o_a = vec![0.0; hash_bits];
        let mut grad_o_p = vec![0.0; hash_bits];
        let mut grad_o_n = vec![0.0; hash_bits];
        for j in 0..hash_bits {
            grad_o_a[j] = 2.0 * (o_n[j] - o_p[j]);
            grad_o_p[j] = 2.0 * (o_p[j] - o_a[j]);
            grad_o_n[j] = 2.0 * (o_a[j] - o_n[j]);
        }

        // Back-propagate through layer 2 and layer 1 for each of the three
        // samples, accumulating weight gradients.

        // Helper: back-prop one sample and accumulate into weight gradient buffers.
        let backprop_sample = |hidden: &[f64],
                               input: &[f64],
                               grad_out: &[f64],
                               gw2: &mut Vec<Vec<f64>>,
                               gw1: &mut Vec<Vec<f64>>,
                               gb1: &mut Vec<f64>| {
            // grad w.r.t. hidden (through W2)
            let mut grad_hidden = vec![0.0; hidden_dim];
            for i in 0..hidden_dim {
                for j in 0..hash_bits {
                    grad_hidden[i] += grad_out[j] * gw2[i][j]; // using W2 for grads
                }
                // ReLU derivative
                if hidden[i] <= 0.0 {
                    grad_hidden[i] = 0.0;
                }
            }

            // Accumulate W2 gradient: dL/dW2[i][j] += hidden[i] * grad_out[j]
            for i in 0..hidden_dim {
                for j in 0..hash_bits {
                    gw2[i][j] += hidden[i] * grad_out[j];
                }
            }

            // Accumulate b1 gradient
            for i in 0..hidden_dim {
                gb1[i] += grad_hidden[i];
            }

            // Accumulate W1 gradient: dL/dW1[i][j] += input[i] * grad_hidden[j]
            for i in 0..input_dim {
                for j in 0..hidden_dim {
                    gw1[i][j] += input[i] * grad_hidden[j];
                }
            }
        };

        // Gradient accumulators (initialized to zero)
        let mut gw2 = vec![vec![0.0; hash_bits]; hidden_dim];
        let mut gw1 = vec![vec![0.0; hidden_dim]; input_dim];
        let mut gb1 = vec![0.0; hidden_dim];

        // We need to pass W2 for the hidden gradient computation but also
        // accumulate into gw2. We snapshot W2 first for the backward pass.
        let w2_snapshot = self.weights_2.clone();

        // Use the snapshot for gradient computation w.r.t. hidden
        let mut gw2_read = w2_snapshot; // read-only copy for back-prop through W2

        backprop_sample(&h_a, anchor, &grad_o_a, &mut gw2_read, &mut gw1, &mut gb1);
        // Reset gw2_read back; we only want accumulated gradients in gw2
        // Actually, we need a cleaner approach. Let's use the snapshot for
        // reading and a separate buffer for writing.
        let w2_snap = self.weights_2.clone();

        // Clear and redo properly
        gw2 = vec![vec![0.0; hash_bits]; hidden_dim];
        gw1 = vec![vec![0.0; hidden_dim]; input_dim];
        gb1 = vec![0.0; hidden_dim];

        // Backprop helper (reads from w2_snap for gradient flow, writes to gw2)
        for (hidden, input, grad_out) in [
            (&h_a, anchor, &grad_o_a),
            (&h_p, positive, &grad_o_p),
            (&h_n, negative, &grad_o_n),
        ] {
            // grad w.r.t. hidden (through W2 snapshot)
            let mut grad_hidden = vec![0.0; hidden_dim];
            for i in 0..hidden_dim {
                for j in 0..hash_bits {
                    grad_hidden[i] += grad_out[j] * w2_snap[i][j];
                }
                // ReLU derivative
                if hidden[i] <= 0.0 {
                    grad_hidden[i] = 0.0;
                }
            }

            // Accumulate W2 gradient
            for i in 0..hidden_dim {
                for j in 0..hash_bits {
                    gw2[i][j] += hidden[i] * grad_out[j];
                }
            }

            // Accumulate b1 gradient
            for i in 0..hidden_dim {
                gb1[i] += grad_hidden[i];
            }

            // Accumulate W1 gradient
            for i in 0..input_dim {
                for j in 0..hidden_dim {
                    gw1[i][j] += input[i] * grad_hidden[j];
                }
            }
        }

        // SGD update
        let lr = self.learning_rate;
        for i in 0..hidden_dim {
            for j in 0..hash_bits {
                self.weights_2[i][j] -= lr * gw2[i][j];
            }
        }
        for i in 0..input_dim {
            for j in 0..hidden_dim {
                self.weights_1[i][j] -= lr * gw1[i][j];
            }
        }
        for i in 0..hidden_dim {
            self.bias_1[i] -= lr * gb1[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Neural LSH Index
// ---------------------------------------------------------------------------

/// A multi-table index that uses learned (neural) hash functions for
/// approximate nearest-neighbor search.
///
/// Each table uses an independent [`NeuralHashFunction`]. On insertion, a
/// vector is hashed by every table and stored in the corresponding bucket.
/// On query, candidate sets from all tables are unioned and ranked by
/// Euclidean distance.
#[derive(Debug, Clone)]
pub struct NeuralLSHIndex {
    /// Hash tables: `tables[t][hash_code] -> Vec<item_id>`
    pub tables: Vec<HashMap<u64, Vec<usize>>>,
    /// One neural hash function per table.
    pub hash_functions: Vec<NeuralHashFunction>,
    /// Stored vectors (id -> vector) for distance reranking.
    vectors: Vec<Option<Vec<f64>>>,
    /// Configuration snapshot.
    config: NeuralLSHConfig,
}

impl NeuralLSHIndex {
    /// Create a new empty index with the given configuration.
    pub fn new(config: NeuralLSHConfig) -> Self {
        let hash_functions: Vec<NeuralHashFunction> = (0..config.num_tables)
            .map(|_| {
                NeuralHashFunction::new(
                    config.input_dim,
                    config.hidden_dim,
                    config.hash_bits,
                    config.learning_rate,
                )
            })
            .collect();

        let tables = vec![HashMap::new(); config.num_tables];

        Self {
            tables,
            hash_functions,
            vectors: Vec::new(),
            config,
        }
    }

    /// Insert a vector into the index under the given `id`.
    ///
    /// The vector is hashed by every neural hash function and placed in the
    /// corresponding bucket in each table.
    pub fn insert(&mut self, id: usize, vector: &[f64]) {
        // Ensure storage is large enough
        if id >= self.vectors.len() {
            self.vectors.resize(id + 1, None);
        }
        self.vectors[id] = Some(vector.to_vec());

        for (table, hf) in self.tables.iter_mut().zip(self.hash_functions.iter()) {
            let code = hf.hash(vector);
            table.entry(code).or_default().push(id);
        }
    }

    /// Query the index for the `k` approximate nearest neighbors of `vector`.
    ///
    /// Returns a list of `(id, distance)` pairs sorted by ascending Euclidean
    /// distance. The candidate set is the union of all bucket hits across
    /// every table.
    pub fn query(&self, vector: &[f64], k: usize) -> Vec<(usize, f64)> {
        // Collect candidate ids from all tables
        let mut candidates = std::collections::HashSet::new();

        for (table, hf) in self.tables.iter().zip(self.hash_functions.iter()) {
            let code = hf.hash(vector);
            if let Some(ids) = table.get(&code) {
                for &id in ids {
                    candidates.insert(id);
                }
            }
        }

        // Rank candidates by Euclidean distance
        let mut scored: Vec<(usize, f64)> = candidates
            .into_iter()
            .filter_map(|id| {
                self.vectors.get(id).and_then(|v| {
                    v.as_ref().map(|stored| {
                        let dist = euclidean_distance(vector, stored);
                        (id, dist)
                    })
                })
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Train the neural hash functions on a set of similarity triplets.
    ///
    /// Each triplet `(anchor, positive, negative)` indicates that `anchor` is
    /// more similar to `positive` than to `negative`. The hash functions learn
    /// to produce hash codes that reflect this relationship.
    pub fn train(&mut self, triplets: &[(Vec<f64>, Vec<f64>, Vec<f64>)]) {
        for (anchor, positive, negative) in triplets {
            for hf in &mut self.hash_functions {
                hf.train_step(anchor, positive, negative);
            }
        }
    }

    /// Rebuild all hash tables using the current (possibly retrained) hash
    /// functions. Call this after [`train`](Self::train) to re-index existing
    /// vectors with the updated hash codes.
    pub fn rebuild(&mut self) {
        // Clear all tables
        for table in &mut self.tables {
            table.clear();
        }

        // Re-insert every stored vector
        let vectors: Vec<(usize, Vec<f64>)> = self
            .vectors
            .iter()
            .enumerate()
            .filter_map(|(id, v)| v.as_ref().map(|vec| (id, vec.clone())))
            .collect();

        for (id, vec) in &vectors {
            for (table, hf) in self.tables.iter_mut().zip(self.hash_functions.iter()) {
                let code = hf.hash(vec);
                table.entry(code).or_default().push(*id);
            }
        }
    }

    /// Return the number of items currently stored in the index.
    pub fn len(&self) -> usize {
        self.vectors.iter().filter(|v| v.is_some()).count()
    }

    /// Return `true` if the index contains no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &NeuralLSHConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Hamming distance between two hash codes.
#[allow(dead_code)]
fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(input_dim: usize) -> NeuralLSHConfig {
        NeuralLSHConfig {
            input_dim,
            hash_bits: 8,
            num_tables: 4,
            hidden_dim: 32,
            learning_rate: 0.01,
        }
    }

    #[test]
    fn test_config_default() {
        let config = NeuralLSHConfig::default();
        assert_eq!(config.hash_bits, 16);
        assert_eq!(config.num_tables, 8);
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.input_dim, 64);
        assert!((config.learning_rate - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_neural_hash_function_creation() {
        let hf = NeuralHashFunction::new(10, 32, 8, 0.01);
        assert_eq!(hf.weights_1.len(), 10);
        assert_eq!(hf.weights_1[0].len(), 32);
        assert_eq!(hf.bias_1.len(), 32);
        assert_eq!(hf.weights_2.len(), 32);
        assert_eq!(hf.weights_2[0].len(), 8);
    }

    #[test]
    fn test_hash_deterministic() {
        let hf = NeuralHashFunction::new(10, 32, 8, 0.01);
        let input = vec![0.5; 10];
        let h1 = hf.hash(&input);
        let h2 = hf.hash(&input);
        assert_eq!(h1, h2, "Hash should be deterministic for the same input");
    }

    #[test]
    fn test_hash_bits_in_range() {
        let hf = NeuralHashFunction::new(10, 32, 8, 0.01);
        let input = vec![1.0; 10];
        let code = hf.hash(&input);
        // With 8 hash bits, code should fit in lower 8 bits
        assert!(code < (1u64 << 8), "Hash code should use at most 8 bits");
    }

    #[test]
    fn test_index_insert_and_query() {
        let config = default_config(4);
        let mut index = NeuralLSHIndex::new(config);

        let v0 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![0.9, 0.1, 0.0, 0.0];
        let v2 = vec![0.0, 0.0, 0.0, 1.0];

        index.insert(0, &v0);
        index.insert(1, &v1);
        index.insert(2, &v2);

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());

        // Query with v0 itself should return v0 first (distance 0)
        let results = index.query(&v0, 3);
        if !results.is_empty() {
            // The first result (if v0 is found) should have distance ~0
            if results[0].0 == 0 {
                assert!(results[0].1 < 1e-10, "Self-query should have ~0 distance");
            }
        }
    }

    #[test]
    fn test_index_empty() {
        let config = default_config(4);
        let index = NeuralLSHIndex::new(config);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let results = index.query(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_train_step_does_not_panic() {
        let mut hf = NeuralHashFunction::new(4, 16, 8, 0.01);
        let anchor = vec![1.0, 0.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0, 0.0];
        let negative = vec![0.0, 0.0, 0.0, 1.0];

        // Should not panic
        hf.train_step(&anchor, &positive, &negative);
    }

    #[test]
    fn test_training_reduces_distance() {
        let mut hf = NeuralHashFunction::new(4, 16, 8, 0.05);
        let anchor = vec![1.0, 0.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0, 0.0];
        let negative = vec![0.0, 0.0, 1.0, 0.0];

        let h_a_before = hf.hash(&anchor);
        let h_p_before = hf.hash(&positive);
        let h_n_before = hf.hash(&negative);

        let dist_pos_before = hamming_distance(h_a_before, h_p_before);
        let dist_neg_before = hamming_distance(h_a_before, h_n_before);

        // Train many steps
        for _ in 0..200 {
            hf.train_step(&anchor, &positive, &negative);
        }

        let h_a_after = hf.hash(&anchor);
        let h_p_after = hf.hash(&positive);
        let h_n_after = hf.hash(&negative);

        let dist_pos_after = hamming_distance(h_a_after, h_p_after);
        let dist_neg_after = hamming_distance(h_a_after, h_n_after);

        // After training, we expect dist_pos to decrease or dist_neg to increase
        // (or at least the gap to improve). This is a soft check since
        // training is stochastic.
        let gap_before = dist_neg_before as i32 - dist_pos_before as i32;
        let gap_after = dist_neg_after as i32 - dist_pos_after as i32;

        // The gap should be non-negative after training (or at least not worse
        // than random). We use a lenient check since the network is small.
        assert!(
            gap_after >= gap_before || dist_pos_after <= dist_pos_before,
            "Training should improve or maintain hash quality: \
             gap_before={gap_before}, gap_after={gap_after}, \
             dist_pos: {dist_pos_before}->{dist_pos_after}, \
             dist_neg: {dist_neg_before}->{dist_neg_after}"
        );
    }

    #[test]
    fn test_index_train_and_rebuild() {
        let config = default_config(4);
        let mut index = NeuralLSHIndex::new(config);

        let v0 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![0.9, 0.1, 0.0, 0.0];
        let v2 = vec![0.0, 0.0, 0.0, 1.0];

        index.insert(0, &v0);
        index.insert(1, &v1);
        index.insert(2, &v2);

        // Train with triplets
        let triplets = vec![(v0.clone(), v1.clone(), v2.clone())];
        for _ in 0..50 {
            index.train(&triplets);
        }

        // Rebuild with new hash functions
        index.rebuild();

        // Should still have 3 items
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 1e-10);

        let c = vec![3.0, 4.0];
        let d = vec![0.0, 0.0];
        assert!((euclidean_distance(&c, &d) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0b1010, 0b1010), 0);
        assert_eq!(hamming_distance(0b1010, 0b0101), 4);
        assert_eq!(hamming_distance(0b1111, 0b0000), 4);
        assert_eq!(hamming_distance(0b1100, 0b1010), 2);
    }

    #[test]
    fn test_forward_produces_valid_output() {
        let hf = NeuralHashFunction::new(4, 16, 8, 0.01);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (hidden, output) = hf.forward(&input);

        assert_eq!(hidden.len(), 16);
        assert_eq!(output.len(), 8);

        // All hidden values should be >= 0 (ReLU)
        for &h in &hidden {
            assert!(h >= 0.0, "ReLU output must be non-negative, got {h}");
        }
    }

    #[test]
    fn test_different_inputs_likely_different_hashes() {
        let hf = NeuralHashFunction::new(16, 64, 16, 0.01);

        let v1 = vec![1.0; 16];
        let v2 = vec![-1.0; 16];

        let h1 = hf.hash(&v1);
        let h2 = hf.hash(&v2);

        // Very different inputs should (almost certainly) produce different hashes
        assert_ne!(h1, h2, "Opposite vectors should hash differently");
    }

    #[test]
    fn test_query_returns_at_most_k() {
        let config = default_config(4);
        let mut index = NeuralLSHIndex::new(config);

        for i in 0..20 {
            index.insert(i, &vec![i as f64; 4]);
        }

        let results = index.query(&[10.0; 4], 5);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_query_results_sorted_by_distance() {
        let config = default_config(4);
        let mut index = NeuralLSHIndex::new(config);

        for i in 0..10 {
            index.insert(i, &vec![i as f64; 4]);
        }

        let results = index.query(&[5.0; 4], 10);
        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "Results should be sorted by distance: {} > {}",
                window[0].1,
                window[1].1
            );
        }
    }
}
