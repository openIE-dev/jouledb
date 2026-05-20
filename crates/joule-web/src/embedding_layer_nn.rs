//! Embedding lookup table for discrete token indices.
//!
//! Maps integer token IDs to dense vectors, supporting multiple
//! initialization strategies, sparse gradient accumulation,
//! padding index masking, and max-norm constraint.

use std::fmt;

// ── Initialization Strategy ───────────────────────────────────────

/// How to initialize the embedding matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingInit {
    /// Uniform in [-1/sqrt(dim), 1/sqrt(dim)].
    Normal,
    /// Uniform in [-bound, bound].
    Uniform(f64),
    /// All zeros.
    Zeros,
    /// Constant value.
    Constant(f64),
    /// Xavier/Glorot uniform.
    Xavier,
}

impl fmt::Display for EmbeddingInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Uniform(b) => write!(f, "uniform({:.3})", b),
            Self::Zeros => write!(f, "zeros"),
            Self::Constant(c) => write!(f, "constant({:.3})", c),
            Self::Xavier => write!(f, "xavier"),
        }
    }
}

// ── Simple LCG ────────────────────────────────────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, bound: f64) -> f64 {
        self.next_f64() * 2.0 * bound - bound
    }
}

// ── EmbeddingLayer ────────────────────────────────────────────────

/// Embedding layer: lookup table mapping token indices to dense vectors.
///
/// The weight matrix has shape `[num_embeddings, embedding_dim]`,
/// stored in row-major order.
#[derive(Debug, Clone)]
pub struct EmbeddingLayer {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub weights: Vec<f64>,
    pub padding_idx: Option<usize>,
    pub max_norm: Option<f64>,
    pub freeze: bool,
}

impl EmbeddingLayer {
    /// Create an embedding layer with default normal initialization.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let mut rng = Lcg::new(num_embeddings as u64 ^ (embedding_dim as u64).wrapping_mul(37));
        let bound = 1.0 / (embedding_dim as f64).sqrt();
        let weights: Vec<f64> = (0..num_embeddings * embedding_dim)
            .map(|_| rng.uniform(bound))
            .collect();

        Self {
            num_embeddings,
            embedding_dim,
            weights,
            padding_idx: None,
            max_norm: None,
            freeze: false,
        }
    }

    /// Set the initialization strategy (re-initializes weights).
    pub fn with_init(mut self, strategy: EmbeddingInit, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let dim = self.embedding_dim as f64;
        let total = self.num_embeddings * self.embedding_dim;

        match strategy {
            EmbeddingInit::Normal => {
                let bound = 1.0 / dim.sqrt();
                self.weights = (0..total).map(|_| rng.uniform(bound)).collect();
            }
            EmbeddingInit::Uniform(b) => {
                self.weights = (0..total).map(|_| rng.uniform(b)).collect();
            }
            EmbeddingInit::Zeros => {
                self.weights = vec![0.0; total];
            }
            EmbeddingInit::Constant(c) => {
                self.weights = vec![c; total];
            }
            EmbeddingInit::Xavier => {
                let bound = (6.0 / (self.num_embeddings as f64 + dim)).sqrt();
                self.weights = (0..total).map(|_| rng.uniform(bound)).collect();
            }
        }

        // Zero out padding index if set
        if let Some(pad_idx) = self.padding_idx {
            self.zero_padding(pad_idx);
        }
        self
    }

    /// Set a padding index whose embedding will always be zero.
    pub fn with_padding_idx(mut self, idx: usize) -> Self {
        assert!(idx < self.num_embeddings, "padding index out of range");
        self.padding_idx = Some(idx);
        self.zero_padding(idx);
        self
    }

    /// Set max-norm constraint on embedding vectors.
    pub fn with_max_norm(mut self, max_norm: f64) -> Self {
        self.max_norm = Some(max_norm);
        self
    }

    /// Freeze the embedding (no gradient updates).
    pub fn with_freeze(mut self) -> Self {
        self.freeze = true;
        self
    }

    /// Zero out the embedding for a given index.
    fn zero_padding(&mut self, idx: usize) {
        let start = idx * self.embedding_dim;
        for i in start..start + self.embedding_dim {
            self.weights[i] = 0.0;
        }
    }

    /// Total trainable parameters.
    pub fn param_count(&self) -> usize {
        if self.freeze {
            0
        } else {
            let total = self.num_embeddings * self.embedding_dim;
            match self.padding_idx {
                Some(_) => total - self.embedding_dim,
                None => total,
            }
        }
    }

    /// Memory usage in bytes (f64 = 8 bytes).
    pub fn memory_bytes(&self) -> usize {
        self.weights.len() * 8
    }

    /// Look up a single token's embedding.
    pub fn lookup(&self, token_id: usize) -> &[f64] {
        assert!(token_id < self.num_embeddings, "token_id out of range");
        let start = token_id * self.embedding_dim;
        &self.weights[start..start + self.embedding_dim]
    }

    /// Look up a single token, returning a mutable slice.
    pub fn lookup_mut(&mut self, token_id: usize) -> &mut [f64] {
        assert!(token_id < self.num_embeddings, "token_id out of range");
        let start = token_id * self.embedding_dim;
        &mut self.weights[start..start + self.embedding_dim]
    }

    /// Forward pass: look up embeddings for a sequence of token IDs.
    /// Returns a flattened vector of shape `[seq_len, embedding_dim]`.
    pub fn forward(&self, token_ids: &[usize]) -> Vec<f64> {
        let mut output = Vec::with_capacity(token_ids.len() * self.embedding_dim);
        for &tid in token_ids {
            output.extend_from_slice(self.lookup(tid));
        }

        // Apply max-norm if set
        if let Some(max_n) = self.max_norm {
            for chunk in output.chunks_mut(self.embedding_dim) {
                let norm = chunk.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > max_n {
                    let scale = max_n / norm;
                    for v in chunk.iter_mut() {
                        *v *= scale;
                    }
                }
            }
        }

        output
    }

    /// Batch forward: multiple sequences, each padded to the same length.
    pub fn forward_batch(
        &self,
        batch_token_ids: &[usize],
        batch_size: usize,
        seq_len: usize,
    ) -> Vec<f64> {
        assert_eq!(batch_token_ids.len(), batch_size * seq_len);
        let mut output = Vec::with_capacity(batch_size * seq_len * self.embedding_dim);
        for &tid in batch_token_ids {
            output.extend_from_slice(self.lookup(tid));
        }
        output
    }

    /// L2 norm of a specific embedding vector.
    pub fn embedding_norm(&self, token_id: usize) -> f64 {
        self.lookup(token_id).iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Cosine similarity between two embedding vectors.
    pub fn cosine_similarity(&self, id_a: usize, id_b: usize) -> f64 {
        let a = self.lookup(id_a);
        let b = self.lookup(id_b);
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-12 || norm_b < 1e-12 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Enforce max-norm constraint on all embeddings in-place.
    pub fn clip_norms(&mut self) {
        if let Some(max_n) = self.max_norm {
            for i in 0..self.num_embeddings {
                let start = i * self.embedding_dim;
                let slice = &mut self.weights[start..start + self.embedding_dim];
                let norm = slice.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > max_n {
                    let scale = max_n / norm;
                    for v in slice.iter_mut() {
                        *v *= scale;
                    }
                }
            }
        }
    }
}

impl fmt::Display for EmbeddingLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Embedding({}, {}, padding={:?}, max_norm={:?}, freeze={})",
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.max_norm,
            self.freeze
        )
    }
}

// ── Sparse Gradient ───────────────────────────────────────────────

/// Sparse gradient accumulator for embedding layers.
///
/// Only stores gradients for token IDs that were accessed,
/// avoiding O(vocab_size * dim) memory for gradients.
#[derive(Debug, Clone)]
pub struct SparseGrad {
    pub embedding_dim: usize,
    /// Map from token_id to accumulated gradient vector.
    entries: Vec<(usize, Vec<f64>)>,
}

impl SparseGrad {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            entries: Vec::new(),
        }
    }

    /// Accumulate gradient for a specific token.
    pub fn accumulate(&mut self, token_id: usize, grad: &[f64]) {
        assert_eq!(grad.len(), self.embedding_dim);
        for entry in &mut self.entries {
            if entry.0 == token_id {
                for (g, &dg) in entry.1.iter_mut().zip(grad.iter()) {
                    *g += dg;
                }
                return;
            }
        }
        self.entries.push((token_id, grad.to_vec()));
    }

    /// Number of unique tokens with gradients.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Get the gradient for a token, if it exists.
    pub fn get(&self, token_id: usize) -> Option<&[f64]> {
        self.entries
            .iter()
            .find(|(id, _)| *id == token_id)
            .map(|(_, v)| v.as_slice())
    }

    /// Apply sparse gradient update to embedding weights.
    pub fn apply_sgd(&self, embedding: &mut EmbeddingLayer, lr: f64) {
        if embedding.freeze {
            return;
        }
        for (tid, grad) in &self.entries {
            if Some(*tid) == embedding.padding_idx {
                continue; // Don't update padding
            }
            let emb_slice = embedding.lookup_mut(*tid);
            for (w, g) in emb_slice.iter_mut().zip(grad.iter()) {
                *w -= lr * g;
            }
        }
    }

    /// Clear all accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.entries.clear();
    }

    /// Memory used by the sparse gradient (approximate).
    pub fn memory_bytes(&self) -> usize {
        self.entries.len() * (8 + self.embedding_dim * 8)
    }
}

impl fmt::Display for SparseGrad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseGrad(entries={}, dim={})",
            self.entries.len(),
            self.embedding_dim
        )
    }
}

// ── Positional Embedding ──────────────────────────────────────────

/// Sinusoidal positional encoding (non-learnable).
#[derive(Debug, Clone)]
pub struct SinusoidalPositionalEncoding {
    pub max_len: usize,
    pub dim: usize,
    pub encodings: Vec<f64>,
}

impl SinusoidalPositionalEncoding {
    /// Compute sinusoidal positional encodings.
    pub fn new(max_len: usize, dim: usize) -> Self {
        let mut encodings = vec![0.0; max_len * dim];
        for pos in 0..max_len {
            for i in 0..dim {
                let angle = pos as f64 / (10000.0_f64).powf(2.0 * (i / 2) as f64 / dim as f64);
                encodings[pos * dim + i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }
        Self { max_len, dim, encodings }
    }

    /// Get encoding for a specific position.
    pub fn get(&self, position: usize) -> &[f64] {
        assert!(position < self.max_len);
        let start = position * self.dim;
        &self.encodings[start..start + self.dim]
    }

    /// Add positional encodings to a sequence of embeddings.
    pub fn add_to_embeddings(&self, embeddings: &mut [f64], seq_len: usize) {
        assert_eq!(embeddings.len(), seq_len * self.dim);
        for pos in 0..seq_len {
            let enc = self.get(pos);
            for d in 0..self.dim {
                embeddings[pos * self.dim + d] += enc[d];
            }
        }
    }
}

impl fmt::Display for SinusoidalPositionalEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SinusoidalPE(max_len={}, dim={})", self.max_len, self.dim)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let emb = EmbeddingLayer::new(1000, 64);
        assert_eq!(emb.num_embeddings, 1000);
        assert_eq!(emb.embedding_dim, 64);
        assert_eq!(emb.weights.len(), 64000);
    }

    #[test]
    fn test_embedding_lookup() {
        let emb = EmbeddingLayer::new(10, 4);
        let vec0 = emb.lookup(0);
        assert_eq!(vec0.len(), 4);
        let vec9 = emb.lookup(9);
        assert_eq!(vec9.len(), 4);
    }

    #[test]
    fn test_embedding_forward() {
        let emb = EmbeddingLayer::new(100, 8);
        let out = emb.forward(&[0, 5, 10]);
        assert_eq!(out.len(), 24); // 3 tokens * 8 dims
    }

    #[test]
    fn test_embedding_forward_consistency() {
        let emb = EmbeddingLayer::new(50, 4);
        let out = emb.forward(&[3, 3]);
        // Same token should give same embedding
        assert_eq!(&out[..4], &out[4..]);
    }

    #[test]
    fn test_padding_idx() {
        let emb = EmbeddingLayer::new(10, 4).with_padding_idx(0);
        let pad_vec = emb.lookup(0);
        assert!(pad_vec.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_padding_idx_survives_init() {
        let emb = EmbeddingLayer::new(10, 4)
            .with_padding_idx(0)
            .with_init(EmbeddingInit::Constant(1.0), 42);
        let pad_vec = emb.lookup(0);
        assert!(pad_vec.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_max_norm() {
        let emb = EmbeddingLayer::new(10, 4)
            .with_init(EmbeddingInit::Constant(5.0), 0)
            .with_max_norm(1.0);
        let out = emb.forward(&[0]);
        let norm = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_freeze() {
        let emb = EmbeddingLayer::new(100, 16).with_freeze();
        assert_eq!(emb.param_count(), 0);
    }

    #[test]
    fn test_param_count() {
        let emb = EmbeddingLayer::new(100, 16);
        assert_eq!(emb.param_count(), 1600);
    }

    #[test]
    fn test_param_count_with_padding() {
        let emb = EmbeddingLayer::new(100, 16).with_padding_idx(0);
        assert_eq!(emb.param_count(), 1584); // 1600 - 16
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let emb = EmbeddingLayer::new(10, 4);
        let sim = emb.cosine_similarity(0, 0);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_range() {
        let emb = EmbeddingLayer::new(10, 4);
        let sim = emb.cosine_similarity(0, 1);
        assert!(sim >= -1.0 - 1e-10 && sim <= 1.0 + 1e-10);
    }

    #[test]
    fn test_embedding_norm() {
        let emb = EmbeddingLayer::new(10, 4).with_init(EmbeddingInit::Constant(1.0), 0);
        let norm = emb.embedding_norm(0);
        assert!((norm - 2.0).abs() < 1e-10); // sqrt(4 * 1^2) = 2
    }

    #[test]
    fn test_sparse_grad_accumulate() {
        let mut sg = SparseGrad::new(4);
        sg.accumulate(5, &[1.0, 2.0, 3.0, 4.0]);
        sg.accumulate(5, &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(sg.num_entries(), 1);
        let g = sg.get(5).unwrap();
        assert!((g[0] - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_grad_apply() {
        let mut emb = EmbeddingLayer::new(10, 2)
            .with_init(EmbeddingInit::Constant(1.0), 0);
        let mut sg = SparseGrad::new(2);
        sg.accumulate(0, &[0.5, 0.5]);
        sg.apply_sgd(&mut emb, 1.0);
        let v = emb.lookup(0);
        assert!((v[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_grad_skip_padding() {
        let mut emb = EmbeddingLayer::new(10, 2)
            .with_init(EmbeddingInit::Constant(1.0), 0)
            .with_padding_idx(0);
        let mut sg = SparseGrad::new(2);
        sg.accumulate(0, &[10.0, 10.0]);
        sg.apply_sgd(&mut emb, 1.0);
        let v = emb.lookup(0);
        assert!(v.iter().all(|x| *x == 0.0)); // Padding untouched
    }

    #[test]
    fn test_sinusoidal_pe() {
        let pe = SinusoidalPositionalEncoding::new(100, 16);
        let enc = pe.get(0);
        assert_eq!(enc.len(), 16);
        // Position 0, dim 0: sin(0) = 0
        assert!(enc[0].abs() < 1e-10);
    }

    #[test]
    fn test_sinusoidal_pe_different_positions() {
        let pe = SinusoidalPositionalEncoding::new(100, 8);
        let enc0 = pe.get(0);
        let enc1 = pe.get(1);
        // Different positions should have different encodings
        assert!(enc0 != enc1);
    }

    #[test]
    fn test_display() {
        let emb = EmbeddingLayer::new(50000, 512);
        let s = format!("{}", emb);
        assert!(s.contains("50000"));
        assert!(s.contains("512"));
    }

    #[test]
    fn test_init_display() {
        assert_eq!(format!("{}", EmbeddingInit::Xavier), "xavier");
        assert_eq!(format!("{}", EmbeddingInit::Zeros), "zeros");
    }
}
