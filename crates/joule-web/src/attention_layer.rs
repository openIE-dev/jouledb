//! Scaled dot-product attention and multi-head attention layer.
//!
//! Implements the core attention mechanism from "Attention Is All You
//! Need" (Vaswani et al., 2017), including query-key-value projections,
//! scaled dot-product scoring, causal (autoregressive) masking,
//! padding masks, and attention weight output.

use std::fmt;

// ── Softmax Utility ───────────────────────────────────────────────

/// Numerically stable softmax over a slice (in-place).
fn softmax_inplace(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in values.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in values.iter_mut() {
            *v /= sum;
        }
    }
}

/// Softmax returning a new vector.
fn softmax_vec(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    softmax_inplace(&mut out);
    out
}

// ── Attention Mask ────────────────────────────────────────────────

/// Types of attention masks.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionMask {
    /// No masking.
    None,
    /// Causal (lower-triangular) mask: position i can only attend to <= i.
    Causal(usize),
    /// Explicit boolean mask: `true` = attend, `false` = masked out.
    /// Shape: `[query_len, key_len]` in row-major order.
    Explicit(Vec<bool>, usize, usize),
    /// Padding mask: `true` = valid position, `false` = padding.
    /// Shape: `[seq_len]`.
    Padding(Vec<bool>),
}

impl AttentionMask {
    /// Create a causal mask for a given sequence length.
    pub fn causal(seq_len: usize) -> Self {
        Self::Causal(seq_len)
    }

    /// Create a padding mask from sequence lengths.
    pub fn from_lengths(lengths: &[usize], max_len: usize) -> Vec<Self> {
        lengths
            .iter()
            .map(|len| {
                let mask: Vec<bool> = (0..max_len).map(|i| i < *len).collect();
                Self::Padding(mask)
            })
            .collect()
    }

    /// Check if a (query_pos, key_pos) pair is allowed.
    pub fn is_allowed(&self, query_pos: usize, key_pos: usize) -> bool {
        match self {
            Self::None => true,
            Self::Causal(_) => key_pos <= query_pos,
            Self::Explicit(mask, _qlen, klen) => mask[query_pos * klen + key_pos],
            Self::Padding(mask) => mask[key_pos],
        }
    }

    /// Generate the full score mask matrix (NEG_INFINITY for masked positions).
    pub fn to_score_mask(&self, query_len: usize, key_len: usize) -> Vec<f64> {
        let mut mask = vec![0.0; query_len * key_len];
        for q in 0..query_len {
            for k in 0..key_len {
                if !self.is_allowed(q, k) {
                    mask[q * key_len + k] = f64::NEG_INFINITY;
                }
            }
        }
        mask
    }
}

impl fmt::Display for AttentionMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "no_mask"),
            Self::Causal(n) => write!(f, "causal({})", n),
            Self::Explicit(_, q, k) => write!(f, "explicit({}x{})", q, k),
            Self::Padding(m) => write!(f, "padding(len={})", m.len()),
        }
    }
}

// ── Scaled Dot-Product Attention ──────────────────────────────────

/// Result of an attention computation.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Output values: `[query_len, value_dim]` flattened.
    pub values: Vec<f64>,
    /// Attention weights: `[query_len, key_len]` flattened.
    pub weights: Vec<f64>,
    pub query_len: usize,
    pub key_len: usize,
    pub value_dim: usize,
}

impl AttentionOutput {
    /// Get attention weights for a specific query position.
    pub fn weights_for_query(&self, query_pos: usize) -> &[f64] {
        let start = query_pos * self.key_len;
        &self.weights[start..start + self.key_len]
    }

    /// Get the output vector for a specific query position.
    pub fn value_for_query(&self, query_pos: usize) -> &[f64] {
        let start = query_pos * self.value_dim;
        &self.values[start..start + self.value_dim]
    }

    /// Entropy of attention weights per query position (measures focus).
    pub fn attention_entropy(&self) -> Vec<f64> {
        (0..self.query_len)
            .map(|q| {
                let w = self.weights_for_query(q);
                let mut entropy = 0.0;
                for &p in w {
                    if p > 1e-12 {
                        entropy -= p * p.ln();
                    }
                }
                entropy
            })
            .collect()
    }
}

/// Compute scaled dot-product attention.
///
/// - `queries`: `[query_len, d_k]` flattened
/// - `keys`: `[key_len, d_k]` flattened
/// - `values`: `[key_len, d_v]` flattened
pub fn scaled_dot_product_attention(
    queries: &[f64],
    keys: &[f64],
    values: &[f64],
    d_k: usize,
    d_v: usize,
    mask: &AttentionMask,
) -> AttentionOutput {
    let query_len = queries.len() / d_k;
    let key_len = keys.len() / d_k;
    assert_eq!(values.len(), key_len * d_v);

    let scale = 1.0 / (d_k as f64).sqrt();

    // Compute scores: Q * K^T / sqrt(d_k)
    let mut scores = vec![0.0; query_len * key_len];
    for q in 0..query_len {
        for k in 0..key_len {
            let mut dot = 0.0;
            for d in 0..d_k {
                dot += queries[q * d_k + d] * keys[k * d_k + d];
            }
            scores[q * key_len + k] = dot * scale;
        }
    }

    // Apply mask
    let score_mask = mask.to_score_mask(query_len, key_len);
    for i in 0..scores.len() {
        scores[i] += score_mask[i];
    }

    // Softmax per query
    let mut weights = vec![0.0; query_len * key_len];
    for q in 0..query_len {
        let row_start = q * key_len;
        let row = &scores[row_start..row_start + key_len];
        let sm = softmax_vec(row);
        weights[row_start..row_start + key_len].copy_from_slice(&sm);
    }

    // Weighted sum of values
    let mut output_values = vec![0.0; query_len * d_v];
    for q in 0..query_len {
        for v_dim in 0..d_v {
            let mut sum = 0.0;
            for k in 0..key_len {
                sum += weights[q * key_len + k] * values[k * d_v + v_dim];
            }
            output_values[q * d_v + v_dim] = sum;
        }
    }

    AttentionOutput {
        values: output_values,
        weights,
        query_len,
        key_len,
        value_dim: d_v,
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

    fn uniform(&mut self, bound: f64) -> f64 {
        let val = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        val * 2.0 * bound - bound
    }
}

// ── Linear Projection ─────────────────────────────────────────────

/// A simple linear projection (no bias) used inside multi-head attention.
#[derive(Debug, Clone)]
struct LinearProj {
    in_dim: usize,
    out_dim: usize,
    weights: Vec<f64>,
}

impl LinearProj {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let bound = 1.0 / (in_dim as f64).sqrt();
        let weights: Vec<f64> = (0..in_dim * out_dim)
            .map(|_| rng.uniform(bound))
            .collect();
        Self { in_dim, out_dim, weights }
    }

    /// Project input: `[seq_len, in_dim]` -> `[seq_len, out_dim]`.
    fn forward(&self, input: &[f64], seq_len: usize) -> Vec<f64> {
        let mut output = vec![0.0; seq_len * self.out_dim];
        for s in 0..seq_len {
            for o in 0..self.out_dim {
                let mut sum = 0.0;
                for i in 0..self.in_dim {
                    sum += input[s * self.in_dim + i] * self.weights[o * self.in_dim + i];
                }
                output[s * self.out_dim + o] = sum;
            }
        }
        output
    }

    fn param_count(&self) -> usize {
        self.in_dim * self.out_dim
    }
}

// ── Multi-Head Attention ──────────────────────────────────────────

/// Multi-head attention layer.
///
/// Splits d_model into `num_heads` heads of dimension `d_k = d_model / num_heads`,
/// computes attention independently per head, then concatenates and projects.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_k: usize,
    pub d_v: usize,
    w_q: LinearProj,
    w_k: LinearProj,
    w_v: LinearProj,
    w_o: LinearProj,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        let d_k = d_model / num_heads;
        let d_v = d_k;

        Self {
            d_model,
            num_heads,
            d_k,
            d_v,
            w_q: LinearProj::new(d_model, d_model, 100),
            w_k: LinearProj::new(d_model, d_model, 200),
            w_v: LinearProj::new(d_model, d_model, 300),
            w_o: LinearProj::new(d_model, d_model, 400),
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.w_q = LinearProj::new(self.d_model, self.d_model, seed);
        self.w_k = LinearProj::new(self.d_model, self.d_model, seed + 1);
        self.w_v = LinearProj::new(self.d_model, self.d_model, seed + 2);
        self.w_o = LinearProj::new(self.d_model, self.d_model, seed + 3);
        self
    }

    /// Total trainable parameters.
    pub fn param_count(&self) -> usize {
        self.w_q.param_count()
            + self.w_k.param_count()
            + self.w_v.param_count()
            + self.w_o.param_count()
    }

    /// Forward pass with self-attention.
    ///
    /// `input`: `[seq_len, d_model]` flattened.
    pub fn forward(&self, input: &[f64], seq_len: usize, mask: &AttentionMask) -> AttentionOutput {
        self.forward_cross(input, input, seq_len, seq_len, mask)
    }

    /// Forward pass with cross-attention (queries from one source, keys/values from another).
    pub fn forward_cross(
        &self,
        query_input: &[f64],
        kv_input: &[f64],
        query_len: usize,
        kv_len: usize,
        mask: &AttentionMask,
    ) -> AttentionOutput {
        assert_eq!(query_input.len(), query_len * self.d_model);
        assert_eq!(kv_input.len(), kv_len * self.d_model);

        // Project Q, K, V
        let all_q = self.w_q.forward(query_input, query_len);
        let all_k = self.w_k.forward(kv_input, kv_len);
        let all_v = self.w_v.forward(kv_input, kv_len);

        // Split into heads, compute attention per head, concatenate
        let mut head_outputs = vec![0.0; query_len * self.d_model];
        let mut all_weights = vec![0.0; self.num_heads * query_len * kv_len];

        for h in 0..self.num_heads {
            let head_offset = h * self.d_k;

            // Extract this head's Q, K, V
            let mut q_head = vec![0.0; query_len * self.d_k];
            let mut k_head = vec![0.0; kv_len * self.d_k];
            let mut v_head = vec![0.0; kv_len * self.d_v];

            for s in 0..query_len {
                for d in 0..self.d_k {
                    q_head[s * self.d_k + d] = all_q[s * self.d_model + head_offset + d];
                }
            }
            for s in 0..kv_len {
                for d in 0..self.d_k {
                    k_head[s * self.d_k + d] = all_k[s * self.d_model + head_offset + d];
                    v_head[s * self.d_v + d] = all_v[s * self.d_model + head_offset + d];
                }
            }

            // Attention for this head
            let attn = scaled_dot_product_attention(
                &q_head, &k_head, &v_head, self.d_k, self.d_v, mask,
            );

            // Copy head output into concatenated result
            for s in 0..query_len {
                for d in 0..self.d_v {
                    head_outputs[s * self.d_model + head_offset + d] =
                        attn.values[s * self.d_v + d];
                }
            }

            // Store weights
            let w_offset = h * query_len * kv_len;
            all_weights[w_offset..w_offset + query_len * kv_len]
                .copy_from_slice(&attn.weights);
        }

        // Output projection
        let final_output = self.w_o.forward(&head_outputs, query_len);

        // Average weights across heads for the output
        let mut avg_weights = vec![0.0; query_len * kv_len];
        for h in 0..self.num_heads {
            let w_offset = h * query_len * kv_len;
            for i in 0..query_len * kv_len {
                avg_weights[i] += all_weights[w_offset + i] / self.num_heads as f64;
            }
        }

        AttentionOutput {
            values: final_output,
            weights: avg_weights,
            query_len,
            key_len: kv_len,
            value_dim: self.d_model,
        }
    }

    /// FLOPs for self-attention on a given sequence length.
    pub fn flops(&self, seq_len: usize) -> usize {
        // QKV projections: 3 * seq_len * d_model * d_model * 2
        let proj_flops = 3 * seq_len * self.d_model * self.d_model * 2;
        // Attention scores: num_heads * seq_len * seq_len * d_k * 2
        let score_flops = self.num_heads * seq_len * seq_len * self.d_k * 2;
        // Weighted sum: num_heads * seq_len * seq_len * d_v * 2
        let value_flops = self.num_heads * seq_len * seq_len * self.d_v * 2;
        // Output projection: seq_len * d_model * d_model * 2
        let out_flops = seq_len * self.d_model * self.d_model * 2;
        proj_flops + score_flops + value_flops + out_flops
    }
}

impl fmt::Display for MultiHeadAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiHeadAttention(d_model={}, heads={}, d_k={}, params={})",
            self.d_model,
            self.num_heads,
            self.d_k,
            self.param_count()
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_softmax_basic() {
        let out = softmax_vec(&[1.0, 2.0, 3.0]);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
        assert!(out[2] > out[1]);
    }

    #[test]
    fn test_softmax_inplace_sums_to_one() {
        let mut vals = vec![0.0, 0.0, 0.0];
        softmax_inplace(&mut vals);
        let sum: f64 = vals.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn test_causal_mask() {
        let mask = AttentionMask::causal(4);
        assert!(mask.is_allowed(0, 0));
        assert!(!mask.is_allowed(0, 1));
        assert!(mask.is_allowed(3, 0));
        assert!(mask.is_allowed(3, 3));
    }

    #[test]
    fn test_no_mask() {
        let mask = AttentionMask::None;
        assert!(mask.is_allowed(0, 100));
        assert!(mask.is_allowed(50, 0));
    }

    #[test]
    fn test_padding_mask() {
        let mask = AttentionMask::Padding(vec![true, true, false, false]);
        assert!(mask.is_allowed(0, 0));
        assert!(mask.is_allowed(0, 1));
        assert!(!mask.is_allowed(0, 2));
        assert!(!mask.is_allowed(0, 3));
    }

    #[test]
    fn test_mask_display() {
        assert_eq!(format!("{}", AttentionMask::None), "no_mask");
        assert_eq!(format!("{}", AttentionMask::causal(8)), "causal(8)");
    }

    #[test]
    fn test_score_mask_causal() {
        let mask = AttentionMask::causal(3);
        let sm = mask.to_score_mask(3, 3);
        assert_eq!(sm[0 * 3 + 0], 0.0); // (0,0) allowed
        assert!(sm[0 * 3 + 1].is_infinite()); // (0,1) masked
        assert_eq!(sm[2 * 3 + 1], 0.0); // (2,1) allowed
    }

    #[test]
    fn test_sdp_attention_basic() {
        // 2 queries, 3 keys, d_k=2, d_v=2
        let queries = vec![1.0, 0.0, 0.0, 1.0];
        let keys = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let values = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let out = scaled_dot_product_attention(
            &queries, &keys, &values, 2, 2, &AttentionMask::None,
        );
        assert_eq!(out.values.len(), 4); // 2 queries * 2 value dim
        assert_eq!(out.weights.len(), 6); // 2 queries * 3 keys
    }

    #[test]
    fn test_sdp_weights_sum_to_one() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = scaled_dot_product_attention(&q, &k, &v, 2, 2, &AttentionMask::None);
        let sum: f64 = out.weights.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn test_sdp_causal_mask() {
        // 3 queries, 3 keys
        let q = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = AttentionMask::causal(3);
        let out = scaled_dot_product_attention(&q, &k, &v, 2, 2, &mask);

        // Query 0 can only attend to key 0
        let w0 = out.weights_for_query(0);
        assert!((w0[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_attention_entropy() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 1.0, 0.0]; // 2 identical keys
        let v = vec![1.0, 0.0, 1.0, 0.0];
        let out = scaled_dot_product_attention(&q, &k, &v, 2, 2, &AttentionMask::None);
        let entropy = out.attention_entropy();
        // With 2 identical keys, weights should be uniform => entropy = ln(2)
        assert!((entropy[0] - (2.0_f64).ln()).abs() < EPS);
    }

    #[test]
    fn test_mha_creation() {
        let mha = MultiHeadAttention::new(64, 8);
        assert_eq!(mha.d_k, 8);
        assert_eq!(mha.num_heads, 8);
    }

    #[test]
    fn test_mha_param_count() {
        let mha = MultiHeadAttention::new(64, 8);
        // 4 projection matrices: 4 * 64 * 64 = 16384
        assert_eq!(mha.param_count(), 4 * 64 * 64);
    }

    #[test]
    fn test_mha_forward() {
        let mha = MultiHeadAttention::new(16, 4);
        let input = vec![1.0; 3 * 16]; // seq_len=3, d_model=16
        let out = mha.forward(&input, 3, &AttentionMask::None);
        assert_eq!(out.values.len(), 3 * 16);
        assert_eq!(out.weights.len(), 3 * 3);
    }

    #[test]
    fn test_mha_causal() {
        let mha = MultiHeadAttention::new(8, 2);
        let input = vec![1.0; 4 * 8];
        let mask = AttentionMask::causal(4);
        let out = mha.forward(&input, 4, &mask);
        assert_eq!(out.query_len, 4);
        assert_eq!(out.key_len, 4);
    }

    #[test]
    fn test_mha_cross_attention() {
        let mha = MultiHeadAttention::new(16, 4);
        let q_input = vec![1.0; 2 * 16]; // 2 queries
        let kv_input = vec![1.0; 5 * 16]; // 5 key-value pairs
        let out = mha.forward_cross(&q_input, &kv_input, 2, 5, &AttentionMask::None);
        assert_eq!(out.values.len(), 2 * 16);
        assert_eq!(out.weights.len(), 2 * 5);
    }

    #[test]
    fn test_mha_display() {
        let mha = MultiHeadAttention::new(512, 8);
        let s = format!("{}", mha);
        assert!(s.contains("512"));
        assert!(s.contains("8"));
    }

    #[test]
    fn test_mha_flops() {
        let mha = MultiHeadAttention::new(64, 8);
        let flops = mha.flops(10);
        assert!(flops > 0);
    }

    #[test]
    fn test_from_lengths() {
        let masks = AttentionMask::from_lengths(&[3, 5], 6);
        assert_eq!(masks.len(), 2);
        if let AttentionMask::Padding(ref m) = masks[0] {
            assert_eq!(m, &[true, true, true, false, false, false]);
        } else {
            panic!("expected Padding mask");
        }
    }
}
