//! Positional encoding for transformer models.
//!
//! Implements sinusoidal positional encoding (Vaswani et al. 2017),
//! learned positional embeddings with random initialization, and
//! Rotary Position Embedding (RoPE, Su et al. 2021). All encodings
//! operate on f64 tensors and support configurable sequence lengths
//! and model dimensions.

use std::fmt;

// ── Encoding Type ────────────────────────────────────────────────

/// The type of positional encoding to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EncodingType {
    /// Fixed sinusoidal encoding (sin/cos interleaved).
    Sinusoidal,
    /// Learned embedding table.
    Learned,
    /// Rotary Position Embedding (RoPE).
    Rotary,
}

impl fmt::Display for EncodingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sinusoidal => write!(f, "Sinusoidal"),
            Self::Learned => write!(f, "Learned"),
            Self::Rotary => write!(f, "Rotary"),
        }
    }
}

// ── Configuration ────────────────────────────────────────────────

/// Configuration for positional encoding.
#[derive(Debug, Clone)]
pub struct PosEncodeConfig {
    pub d_model: usize,
    pub max_seq_len: usize,
    pub encoding_type: EncodingType,
    pub base_freq: f64,
    pub dropout_rate: f64,
}

impl PosEncodeConfig {
    pub fn new(d_model: usize, max_seq_len: usize) -> Self {
        Self {
            d_model,
            max_seq_len,
            encoding_type: EncodingType::Sinusoidal,
            base_freq: 10000.0,
            dropout_rate: 0.0,
        }
    }

    pub fn with_encoding_type(mut self, enc: EncodingType) -> Self {
        self.encoding_type = enc;
        self
    }

    pub fn with_base_freq(mut self, freq: f64) -> Self {
        self.base_freq = freq;
        self
    }

    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }
}

// ── Sinusoidal Encoding ──────────────────────────────────────────

/// Precomputed sinusoidal positional encoding table.
///
/// PE(pos, 2i)   = sin(pos / base^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
#[derive(Debug, Clone)]
pub struct SinusoidalEncoding {
    pub config: PosEncodeConfig,
    /// Table of shape (max_seq_len, d_model), row-major.
    pub table: Vec<f64>,
}

impl SinusoidalEncoding {
    /// Precompute the encoding table.
    pub fn new(config: PosEncodeConfig) -> Self {
        let d = config.d_model;
        let max_len = config.max_seq_len;
        let base = config.base_freq;
        let mut table = vec![0.0; max_len * d];

        for pos in 0..max_len {
            for i in 0..(d / 2) {
                let angle = pos as f64 / base.powf(2.0 * i as f64 / d as f64);
                table[pos * d + 2 * i] = angle.sin();
                table[pos * d + 2 * i + 1] = angle.cos();
            }
            // Handle odd d_model
            if d % 2 != 0 {
                let angle = pos as f64 / base.powf(2.0 * (d / 2) as f64 / d as f64);
                table[pos * d + d - 1] = angle.sin();
            }
        }

        Self { config, table }
    }

    /// Get the encoding vector for a specific position.
    pub fn get_position(&self, pos: usize) -> &[f64] {
        let d = self.config.d_model;
        &self.table[pos * d..(pos + 1) * d]
    }

    /// Add positional encoding to input: (seq_len, d_model) -> (seq_len, d_model).
    pub fn encode(&self, input: &[f64], seq_len: usize) -> Vec<f64> {
        let d = self.config.d_model;
        assert_eq!(input.len(), seq_len * d, "input length mismatch");
        assert!(seq_len <= self.config.max_seq_len, "seq_len exceeds max");
        let mut output = input.to_vec();
        for pos in 0..seq_len {
            for dim in 0..d {
                output[pos * d + dim] += self.table[pos * d + dim];
            }
        }
        output
    }

    /// Get the full encoding table as a slice.
    pub fn as_table(&self) -> &[f64] {
        &self.table
    }
}

impl fmt::Display for SinusoidalEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SinusoidalEncoding(d_model={}, max_len={}, base={})",
            self.config.d_model, self.config.max_seq_len, self.config.base_freq
        )
    }
}

// ── Learned Positional Embedding ─────────────────────────────────

/// Learned positional embedding table, initialized with scaled random values.
#[derive(Debug, Clone)]
pub struct LearnedEmbedding {
    pub config: PosEncodeConfig,
    /// Embedding table: (max_seq_len, d_model), row-major.
    pub table: Vec<f64>,
}

impl LearnedEmbedding {
    /// Create with deterministic pseudo-random initialization.
    pub fn new(config: PosEncodeConfig, seed: u64) -> Self {
        let d = config.d_model;
        let max_len = config.max_seq_len;
        let scale = 1.0 / (d as f64).sqrt();
        let mut table = Vec::with_capacity(max_len * d);
        let mut state = seed;

        for _ in 0..(max_len * d) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            table.push((u * 2.0 - 1.0) * scale);
        }

        Self { config, table }
    }

    /// Get the embedding for a specific position.
    pub fn get_position(&self, pos: usize) -> &[f64] {
        let d = self.config.d_model;
        &self.table[pos * d..(pos + 1) * d]
    }

    /// Add learned embeddings to input.
    pub fn encode(&self, input: &[f64], seq_len: usize) -> Vec<f64> {
        let d = self.config.d_model;
        assert_eq!(input.len(), seq_len * d, "input length mismatch");
        assert!(seq_len <= self.config.max_seq_len, "seq_len exceeds max");
        let mut output = input.to_vec();
        for pos in 0..seq_len {
            for dim in 0..d {
                output[pos * d + dim] += self.table[pos * d + dim];
            }
        }
        output
    }

    /// Update a single embedding (simulated gradient step).
    pub fn update_position(&mut self, pos: usize, gradient: &[f64], learning_rate: f64) {
        let d = self.config.d_model;
        assert_eq!(gradient.len(), d, "gradient length must match d_model");
        for dim in 0..d {
            self.table[pos * d + dim] -= learning_rate * gradient[dim];
        }
    }

    /// L2 norm of the embedding at a given position.
    pub fn norm(&self, pos: usize) -> f64 {
        let emb = self.get_position(pos);
        emb.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

impl fmt::Display for LearnedEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LearnedEmbedding(d_model={}, max_len={})",
            self.config.d_model, self.config.max_seq_len
        )
    }
}

// ── Rotary Position Embedding (RoPE) ─────────────────────────────

/// Rotary Position Embedding (RoPE).
///
/// Applies rotation matrices in 2-D subspaces to encode position
/// information directly into Q/K vectors. For each pair of dimensions
/// (2i, 2i+1), the rotation angle is `pos * theta_i` where
/// `theta_i = base^(-2i/d)`.
#[derive(Debug, Clone)]
pub struct RotaryEncoding {
    pub config: PosEncodeConfig,
    /// Precomputed cos(pos * theta) values: (max_seq_len, d_model/2).
    cos_cache: Vec<f64>,
    /// Precomputed sin(pos * theta) values: (max_seq_len, d_model/2).
    sin_cache: Vec<f64>,
}

impl RotaryEncoding {
    /// Create and precompute rotation caches.
    pub fn new(config: PosEncodeConfig) -> Self {
        assert_eq!(config.d_model % 2, 0, "d_model must be even for RoPE");
        let d = config.d_model;
        let half_d = d / 2;
        let max_len = config.max_seq_len;
        let base = config.base_freq;

        let mut cos_cache = Vec::with_capacity(max_len * half_d);
        let mut sin_cache = Vec::with_capacity(max_len * half_d);

        for pos in 0..max_len {
            for i in 0..half_d {
                let theta = pos as f64 / base.powf(2.0 * i as f64 / d as f64);
                cos_cache.push(theta.cos());
                sin_cache.push(theta.sin());
            }
        }

        Self { config, cos_cache, sin_cache }
    }

    /// Apply rotary encoding to a vector at a given position.
    ///
    /// x: slice of length d_model. Returns rotated vector.
    pub fn rotate(&self, x: &[f64], pos: usize) -> Vec<f64> {
        let d = self.config.d_model;
        let half_d = d / 2;
        assert_eq!(x.len(), d, "input must have d_model elements");
        assert!(pos < self.config.max_seq_len, "position out of range");

        let mut out = vec![0.0; d];
        for i in 0..half_d {
            let cos_val = self.cos_cache[pos * half_d + i];
            let sin_val = self.sin_cache[pos * half_d + i];
            // Rotate pair (x[2i], x[2i+1])
            out[2 * i] = x[2 * i] * cos_val - x[2 * i + 1] * sin_val;
            out[2 * i + 1] = x[2 * i] * sin_val + x[2 * i + 1] * cos_val;
        }
        out
    }

    /// Apply rotary encoding to a sequence of vectors: (seq_len, d_model).
    ///
    /// Modifies the input in-place and returns the result.
    pub fn encode(&self, input: &[f64], seq_len: usize) -> Vec<f64> {
        let d = self.config.d_model;
        assert_eq!(input.len(), seq_len * d, "input length mismatch");
        let mut output = Vec::with_capacity(seq_len * d);
        for pos in 0..seq_len {
            let start = pos * d;
            let rotated = self.rotate(&input[start..start + d], pos);
            output.extend_from_slice(&rotated);
        }
        output
    }

    /// Compute the relative attention bias from RoPE.
    ///
    /// The dot product of rotated q and k at positions p and q depends only
    /// on (p - q), giving translation-invariant relative position encoding.
    pub fn relative_bias(&self, pos_q: usize, pos_k: usize) -> f64 {
        let half_d = self.config.d_model / 2;
        let mut bias = 0.0;
        for i in 0..half_d {
            let cos_q = self.cos_cache[pos_q * half_d + i];
            let sin_q = self.sin_cache[pos_q * half_d + i];
            let cos_k = self.cos_cache[pos_k * half_d + i];
            let sin_k = self.sin_cache[pos_k * half_d + i];
            // cos(theta_q - theta_k)
            bias += cos_q * cos_k + sin_q * sin_k;
        }
        bias / half_d as f64
    }
}

impl fmt::Display for RotaryEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RotaryEncoding(d_model={}, max_len={}, base={})",
            self.config.d_model, self.config.max_seq_len, self.config.base_freq
        )
    }
}

// ── Unified Positional Encoder ───────────────────────────────────

/// Unified interface over different positional encoding strategies.
#[derive(Debug, Clone)]
pub enum PositionalEncoder {
    Sinusoidal(SinusoidalEncoding),
    Learned(LearnedEmbedding),
    Rotary(RotaryEncoding),
}

impl PositionalEncoder {
    /// Create from a configuration.
    pub fn from_config(config: PosEncodeConfig, seed: u64) -> Self {
        match config.encoding_type {
            EncodingType::Sinusoidal => Self::Sinusoidal(SinusoidalEncoding::new(config)),
            EncodingType::Learned => Self::Learned(LearnedEmbedding::new(config, seed)),
            EncodingType::Rotary => Self::Rotary(RotaryEncoding::new(config)),
        }
    }

    /// Encode (add positional information to) an input sequence.
    pub fn encode(&self, input: &[f64], seq_len: usize) -> Vec<f64> {
        match self {
            Self::Sinusoidal(enc) => enc.encode(input, seq_len),
            Self::Learned(enc) => enc.encode(input, seq_len),
            Self::Rotary(enc) => enc.encode(input, seq_len),
        }
    }

    /// The encoding type in use.
    pub fn encoding_type(&self) -> EncodingType {
        match self {
            Self::Sinusoidal(_) => EncodingType::Sinusoidal,
            Self::Learned(_) => EncodingType::Learned,
            Self::Rotary(_) => EncodingType::Rotary,
        }
    }
}

impl fmt::Display for PositionalEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sinusoidal(enc) => write!(f, "{}", enc),
            Self::Learned(enc) => write!(f, "{}", enc),
            Self::Rotary(enc) => write!(f, "{}", enc),
        }
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
    fn test_sinusoidal_table_shape() {
        let cfg = PosEncodeConfig::new(16, 100);
        let enc = SinusoidalEncoding::new(cfg);
        assert_eq!(enc.table.len(), 100 * 16);
    }

    #[test]
    fn test_sinusoidal_position_zero() {
        let cfg = PosEncodeConfig::new(4, 10);
        let enc = SinusoidalEncoding::new(cfg);
        let pos0 = enc.get_position(0);
        // sin(0) = 0, cos(0) = 1
        assert!(approx_eq(pos0[0], 0.0, 1e-10));
        assert!(approx_eq(pos0[1], 1.0, 1e-10));
    }

    #[test]
    fn test_sinusoidal_encode_adds() {
        let cfg = PosEncodeConfig::new(4, 10);
        let enc = SinusoidalEncoding::new(cfg);
        let input = vec![1.0; 12]; // 3 positions x 4 dims
        let output = enc.encode(&input, 3);
        // Check that encoding was added (output != input for non-zero positions)
        assert_ne!(output[4], input[4]); // pos=1 should differ
    }

    #[test]
    fn test_sinusoidal_different_positions() {
        let cfg = PosEncodeConfig::new(8, 50);
        let enc = SinusoidalEncoding::new(cfg);
        let p0 = enc.get_position(0);
        let p1 = enc.get_position(1);
        // Positions should produce different encodings
        assert_ne!(p0, p1);
    }

    #[test]
    fn test_sinusoidal_display() {
        let cfg = PosEncodeConfig::new(64, 512);
        let enc = SinusoidalEncoding::new(cfg);
        let s = format!("{}", enc);
        assert!(s.contains("64"));
        assert!(s.contains("512"));
    }

    #[test]
    fn test_learned_embedding_shape() {
        let cfg = PosEncodeConfig::new(8, 32).with_encoding_type(EncodingType::Learned);
        let emb = LearnedEmbedding::new(cfg, 42);
        assert_eq!(emb.table.len(), 32 * 8);
    }

    #[test]
    fn test_learned_embedding_encode() {
        let cfg = PosEncodeConfig::new(4, 10).with_encoding_type(EncodingType::Learned);
        let emb = LearnedEmbedding::new(cfg, 99);
        let input = vec![0.0; 12]; // 3 x 4
        let output = emb.encode(&input, 3);
        // Encoding of zeros should be the embedding itself
        for i in 0..12 {
            assert!(approx_eq(output[i], emb.table[i], 1e-12));
        }
    }

    #[test]
    fn test_learned_embedding_update() {
        let cfg = PosEncodeConfig::new(4, 10).with_encoding_type(EncodingType::Learned);
        let mut emb = LearnedEmbedding::new(cfg, 42);
        let before = emb.get_position(0).to_vec();
        emb.update_position(0, &[1.0, 1.0, 1.0, 1.0], 0.01);
        let after = emb.get_position(0);
        for i in 0..4 {
            assert!(approx_eq(after[i], before[i] - 0.01, 1e-12));
        }
    }

    #[test]
    fn test_learned_embedding_norm() {
        let cfg = PosEncodeConfig::new(4, 10).with_encoding_type(EncodingType::Learned);
        let emb = LearnedEmbedding::new(cfg, 42);
        let n = emb.norm(0);
        assert!(n > 0.0);
    }

    #[test]
    fn test_rope_rotate_identity_at_zero() {
        let cfg = PosEncodeConfig::new(4, 10).with_encoding_type(EncodingType::Rotary);
        let rope = RotaryEncoding::new(cfg);
        let x = vec![1.0, 0.0, 1.0, 0.0];
        let rotated = rope.rotate(&x, 0);
        // At position 0, cos(0)=1 and sin(0)=0, so rotation is identity
        for i in 0..4 {
            assert!(approx_eq(rotated[i], x[i], 1e-10));
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        let cfg = PosEncodeConfig::new(4, 100).with_encoding_type(EncodingType::Rotary);
        let rope = RotaryEncoding::new(cfg);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rotated = rope.rotate(&x, 5);
        let norm_after: f64 = rotated.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Rotation preserves L2 norm
        assert!(approx_eq(norm_before, norm_after, 1e-10));
    }

    #[test]
    fn test_rope_encode_sequence() {
        let cfg = PosEncodeConfig::new(4, 50).with_encoding_type(EncodingType::Rotary);
        let rope = RotaryEncoding::new(cfg);
        let input = vec![1.0; 12]; // 3 positions x 4 dims
        let output = rope.encode(&input, 3);
        assert_eq!(output.len(), 12);
    }

    #[test]
    fn test_rope_relative_bias_self() {
        let cfg = PosEncodeConfig::new(8, 50).with_encoding_type(EncodingType::Rotary);
        let rope = RotaryEncoding::new(cfg);
        let bias = rope.relative_bias(5, 5);
        // Same position -> cos(0) = 1 for all dimensions
        assert!(approx_eq(bias, 1.0, 1e-10));
    }

    #[test]
    fn test_rope_relative_bias_symmetric() {
        let cfg = PosEncodeConfig::new(8, 50).with_encoding_type(EncodingType::Rotary);
        let rope = RotaryEncoding::new(cfg);
        let bias_ab = rope.relative_bias(3, 7);
        let bias_ba = rope.relative_bias(7, 3);
        // cos(a-b) = cos(b-a)
        assert!(approx_eq(bias_ab, bias_ba, 1e-10));
    }

    #[test]
    fn test_unified_encoder_sinusoidal() {
        let cfg = PosEncodeConfig::new(8, 32);
        let enc = PositionalEncoder::from_config(cfg, 0);
        assert_eq!(enc.encoding_type(), EncodingType::Sinusoidal);
        let input = vec![0.0; 24]; // 3 x 8
        let out = enc.encode(&input, 3);
        assert_eq!(out.len(), 24);
    }

    #[test]
    fn test_unified_encoder_learned() {
        let cfg = PosEncodeConfig::new(8, 32).with_encoding_type(EncodingType::Learned);
        let enc = PositionalEncoder::from_config(cfg, 42);
        assert_eq!(enc.encoding_type(), EncodingType::Learned);
    }

    #[test]
    fn test_unified_encoder_rotary() {
        let cfg = PosEncodeConfig::new(8, 32).with_encoding_type(EncodingType::Rotary);
        let enc = PositionalEncoder::from_config(cfg, 0);
        assert_eq!(enc.encoding_type(), EncodingType::Rotary);
    }

    #[test]
    fn test_config_builder() {
        let cfg = PosEncodeConfig::new(64, 512)
            .with_encoding_type(EncodingType::Rotary)
            .with_base_freq(100000.0)
            .with_dropout(0.1);
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.max_seq_len, 512);
        assert!(approx_eq(cfg.base_freq, 100000.0, 1e-6));
        assert!(approx_eq(cfg.dropout_rate, 0.1, 1e-12));
    }

    #[test]
    fn test_encoding_type_display() {
        assert_eq!(format!("{}", EncodingType::Sinusoidal), "Sinusoidal");
        assert_eq!(format!("{}", EncodingType::Learned), "Learned");
        assert_eq!(format!("{}", EncodingType::Rotary), "Rotary");
    }
}
