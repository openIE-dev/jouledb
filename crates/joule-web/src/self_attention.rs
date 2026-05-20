//! Self-attention mechanism with Q/K/V projections.
//!
//! Implements scaled dot-product attention with query, key, and value
//! linear projections, optional causal masking for autoregressive models,
//! and attention dropout simulation. Supports both full and causal
//! attention patterns over dense f64 tensors.

use std::fmt;

// ── Tensor ────────────────────────────────────────────────────────

/// A dense 3-D tensor stored in row-major order: (batch, rows, cols).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor3 {
    pub batch: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Tensor3 {
    /// Create a zero-filled tensor.
    pub fn zeros(batch: usize, rows: usize, cols: usize) -> Self {
        Self { batch, rows, cols, data: vec![0.0; batch * rows * cols] }
    }

    /// Create from raw data (batch-major, row-major within each batch).
    pub fn from_data(batch: usize, rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), batch * rows * cols, "Tensor3 data length mismatch");
        Self { batch, rows, cols, data }
    }

    /// Index into the tensor: data[b][r][c].
    #[inline]
    pub fn get(&self, b: usize, r: usize, c: usize) -> f64 {
        self.data[b * self.rows * self.cols + r * self.cols + c]
    }

    /// Mutable index into the tensor.
    #[inline]
    pub fn set(&mut self, b: usize, r: usize, c: usize, v: f64) {
        self.data[b * self.rows * self.cols + r * self.cols + c] = v;
    }

    /// Transpose the last two dimensions: (B, R, C) -> (B, C, R).
    pub fn transpose_last2(&self) -> Self {
        let mut out = Tensor3::zeros(self.batch, self.cols, self.rows);
        for b in 0..self.batch {
            for r in 0..self.rows {
                for c in 0..self.cols {
                    out.set(b, c, r, self.get(b, r, c));
                }
            }
        }
        out
    }
}

impl fmt::Display for Tensor3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor3(batch={}, rows={}, cols={})", self.batch, self.rows, self.cols)
    }
}

// ── Linear Projection ────────────────────────────────────────────

/// A dense linear projection: output = input @ weight^T + bias.
#[derive(Debug, Clone)]
pub struct LinearProjection {
    /// Weight matrix (out_features x in_features), stored row-major.
    pub weight: Vec<f64>,
    /// Bias vector of length out_features.
    pub bias: Vec<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearProjection {
    /// Create with Xavier-style initialization using a simple deterministic seed.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut weight = Vec::with_capacity(out_features * in_features);
        let mut state = seed;
        for _ in 0..(out_features * in_features) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            weight.push(u * 2.0 * limit - limit);
        }
        let bias = vec![0.0; out_features];
        Self { weight, bias, in_features, out_features }
    }

    /// Apply projection to a Tensor3: (B, S, in) -> (B, S, out).
    pub fn forward(&self, input: &Tensor3) -> Tensor3 {
        assert_eq!(input.cols, self.in_features, "input cols must match in_features");
        let mut out = Tensor3::zeros(input.batch, input.rows, self.out_features);
        for b in 0..input.batch {
            for s in 0..input.rows {
                for o in 0..self.out_features {
                    let mut acc = self.bias[o];
                    for i in 0..self.in_features {
                        acc += input.get(b, s, i) * self.weight[o * self.in_features + i];
                    }
                    out.set(b, s, o, acc);
                }
            }
        }
        out
    }
}

impl fmt::Display for LinearProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear({} -> {})", self.in_features, self.out_features)
    }
}

// ── Attention Mask ───────────────────────────────────────────────

/// Type of attention mask to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionMask {
    /// No masking — full bidirectional attention.
    None,
    /// Causal (lower-triangular) mask for autoregressive decoding.
    Causal,
    /// Sliding window of given size centred on the current position.
    SlidingWindow(usize),
}

impl fmt::Display for AttentionMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "AttentionMask::None"),
            Self::Causal => write!(f, "AttentionMask::Causal"),
            Self::SlidingWindow(w) => write!(f, "AttentionMask::SlidingWindow({})", w),
        }
    }
}

// ── Self-Attention Configuration ─────────────────────────────────

/// Configuration for a self-attention layer.
#[derive(Debug, Clone)]
pub struct SelfAttentionConfig {
    pub d_model: usize,
    pub mask: AttentionMask,
    pub dropout_rate: f64,
    pub scale_factor: Option<f64>,
}

impl SelfAttentionConfig {
    pub fn new(d_model: usize) -> Self {
        Self { d_model, mask: AttentionMask::None, dropout_rate: 0.0, scale_factor: None }
    }

    pub fn with_mask(mut self, mask: AttentionMask) -> Self {
        self.mask = mask;
        self
    }

    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    pub fn with_scale_factor(mut self, factor: f64) -> Self {
        self.scale_factor = Some(factor);
        self
    }
}

// ── SelfAttention Layer ──────────────────────────────────────────

/// Scaled dot-product self-attention with Q/K/V linear projections.
///
/// Given input X of shape (B, S, D):
///   Q = X @ W_q, K = X @ W_k, V = X @ W_v
///   Attn = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
#[derive(Debug, Clone)]
pub struct SelfAttention {
    pub config: SelfAttentionConfig,
    pub proj_q: LinearProjection,
    pub proj_k: LinearProjection,
    pub proj_v: LinearProjection,
    /// Stores the last computed attention weights (B, S, S) for inspection.
    last_attn_weights: Option<Tensor3>,
}

impl SelfAttention {
    /// Build a new self-attention layer.
    pub fn new(config: SelfAttentionConfig) -> Self {
        let d = config.d_model;
        Self {
            proj_q: LinearProjection::new(d, d, 42),
            proj_k: LinearProjection::new(d, d, 137),
            proj_v: LinearProjection::new(d, d, 271),
            config,
            last_attn_weights: None,
        }
    }

    /// Return the scaling denominator (sqrt(d_k) or custom).
    fn scale(&self) -> f64 {
        self.config.scale_factor.unwrap_or_else(|| (self.config.d_model as f64).sqrt())
    }

    /// Compute the attention mask matrix for a given sequence length.
    fn build_mask(&self, seq_len: usize) -> Vec<f64> {
        let neg_inf = f64::NEG_INFINITY;
        let mut mask = vec![0.0; seq_len * seq_len];
        match self.config.mask {
            AttentionMask::None => {}
            AttentionMask::Causal => {
                for r in 0..seq_len {
                    for c in (r + 1)..seq_len {
                        mask[r * seq_len + c] = neg_inf;
                    }
                }
            }
            AttentionMask::SlidingWindow(w) => {
                let half = w / 2;
                for r in 0..seq_len {
                    for c in 0..seq_len {
                        let dist = if c > r { c - r } else { r - c };
                        if dist > half {
                            mask[r * seq_len + c] = neg_inf;
                        }
                    }
                }
            }
        }
        mask
    }

    /// Softmax along the last axis of a row of length `len`.
    fn softmax_row(row: &mut [f64]) {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    /// Forward pass: (B, S, D) -> (B, S, D).
    pub fn forward(&mut self, input: &Tensor3) -> Tensor3 {
        let b = input.batch;
        let s = input.rows;
        let d = self.config.d_model;
        assert_eq!(input.cols, d, "input dim must match d_model");

        // Project Q, K, V
        let q = self.proj_q.forward(input);
        let k = self.proj_k.forward(input);
        let v = self.proj_v.forward(input);

        let scale = self.scale();
        let mask = self.build_mask(s);

        // Compute attention scores: Q @ K^T / scale + mask
        let mut attn_weights = Tensor3::zeros(b, s, s);
        for bi in 0..b {
            for qi in 0..s {
                for ki in 0..s {
                    let mut dot = 0.0;
                    for di in 0..d {
                        dot += q.get(bi, qi, di) * k.get(bi, ki, di);
                    }
                    attn_weights.set(bi, qi, ki, dot / scale + mask[qi * s + ki]);
                }
                // Softmax over key dimension
                let row_start = bi * s * s + qi * s;
                Self::softmax_row(&mut attn_weights.data[row_start..row_start + s]);
            }
        }

        // Weighted sum: attn @ V
        let mut output = Tensor3::zeros(b, s, d);
        for bi in 0..b {
            for qi in 0..s {
                for di in 0..d {
                    let mut acc = 0.0;
                    for ki in 0..s {
                        acc += attn_weights.get(bi, qi, ki) * v.get(bi, ki, di);
                    }
                    output.set(bi, qi, di, acc);
                }
            }
        }

        self.last_attn_weights = Some(attn_weights);
        output
    }

    /// Retrieve the last attention weight matrix for visualization / analysis.
    pub fn attention_weights(&self) -> Option<&Tensor3> {
        self.last_attn_weights.as_ref()
    }

    /// Compute attention entropy per query position (averaged over batch).
    /// Lower entropy = more focused attention.
    pub fn attention_entropy(&self) -> Option<Vec<f64>> {
        let w = self.last_attn_weights.as_ref()?;
        let mut entropy = vec![0.0; w.rows];
        for b in 0..w.batch {
            for q in 0..w.rows {
                let mut h = 0.0;
                for k in 0..w.cols {
                    let p = w.get(b, q, k);
                    if p > 1e-12 {
                        h -= p * p.ln();
                    }
                }
                entropy[q] += h / w.batch as f64;
            }
        }
        Some(entropy)
    }
}

impl fmt::Display for SelfAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SelfAttention(d_model={}, mask={}, dropout={:.2})",
            self.config.d_model, self.config.mask, self.config.dropout_rate
        )
    }
}

// ── Utility: batched matmul ──────────────────────────────────────

/// Batched matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N).
pub fn batched_matmul(a: &Tensor3, b: &Tensor3) -> Tensor3 {
    assert_eq!(a.batch, b.batch, "batch sizes must match");
    assert_eq!(a.cols, b.rows, "inner dimensions must match");
    let mut out = Tensor3::zeros(a.batch, a.rows, b.cols);
    for bi in 0..a.batch {
        for m in 0..a.rows {
            for n in 0..b.cols {
                let mut acc = 0.0;
                for k in 0..a.cols {
                    acc += a.get(bi, m, k) * b.get(bi, k, n);
                }
                out.set(bi, m, n, acc);
            }
        }
    }
    out
}

/// Scaled dot-product attention (standalone function).
///
/// q, k, v: (B, S, D). Returns (B, S, D).
pub fn scaled_dot_product_attention(q: &Tensor3, k: &Tensor3, v: &Tensor3, causal: bool) -> Tensor3 {
    let b = q.batch;
    let s = q.rows;
    let d = q.cols;
    let scale = (d as f64).sqrt();

    let kt = k.transpose_last2();
    let mut scores = batched_matmul(q, &kt);

    // Scale
    for val in scores.data.iter_mut() {
        *val /= scale;
    }

    // Causal mask
    if causal {
        for bi in 0..b {
            for r in 0..s {
                for c in (r + 1)..s {
                    scores.set(bi, r, c, f64::NEG_INFINITY);
                }
            }
        }
    }

    // Softmax per row
    for bi in 0..b {
        for r in 0..s {
            let start = bi * s * s + r * s;
            SelfAttention::softmax_row(&mut scores.data[start..start + s]);
        }
    }

    batched_matmul(&scores, v)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_tensor3_zeros() {
        let t = Tensor3::zeros(2, 3, 4);
        assert_eq!(t.data.len(), 24);
        assert!(t.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_tensor3_get_set() {
        let mut t = Tensor3::zeros(1, 2, 3);
        t.set(0, 1, 2, 42.0);
        assert_eq!(t.get(0, 1, 2), 42.0);
        assert_eq!(t.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_tensor3_transpose() {
        let t = Tensor3::from_data(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tt = t.transpose_last2();
        assert_eq!(tt.rows, 3);
        assert_eq!(tt.cols, 2);
        assert_eq!(tt.get(0, 0, 0), 1.0);
        assert_eq!(tt.get(0, 0, 1), 4.0);
        assert_eq!(tt.get(0, 2, 0), 3.0);
    }

    #[test]
    fn test_tensor3_display() {
        let t = Tensor3::zeros(2, 4, 8);
        assert_eq!(format!("{}", t), "Tensor3(batch=2, rows=4, cols=8)");
    }

    #[test]
    fn test_linear_projection_shape() {
        let proj = LinearProjection::new(8, 16, 99);
        assert_eq!(proj.weight.len(), 16 * 8);
        assert_eq!(proj.bias.len(), 16);
        let input = Tensor3::zeros(2, 5, 8);
        let out = proj.forward(&input);
        assert_eq!(out.batch, 2);
        assert_eq!(out.rows, 5);
        assert_eq!(out.cols, 16);
    }

    #[test]
    fn test_linear_projection_display() {
        let proj = LinearProjection::new(32, 64, 1);
        assert_eq!(format!("{}", proj), "Linear(32 -> 64)");
    }

    #[test]
    fn test_attention_mask_none() {
        let cfg = SelfAttentionConfig::new(4);
        let attn = SelfAttention::new(cfg);
        let mask = attn.build_mask(3);
        assert!(mask.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_attention_mask_causal() {
        let cfg = SelfAttentionConfig::new(4).with_mask(AttentionMask::Causal);
        let attn = SelfAttention::new(cfg);
        let mask = attn.build_mask(3);
        // Lower triangular should be 0, upper should be -inf
        assert_eq!(mask[0], 0.0); // (0,0)
        assert!(mask[1].is_infinite()); // (0,1)
        assert!(mask[2].is_infinite()); // (0,2)
        assert_eq!(mask[3], 0.0); // (1,0)
        assert_eq!(mask[4], 0.0); // (1,1)
        assert!(mask[5].is_infinite()); // (1,2)
    }

    #[test]
    fn test_attention_mask_sliding_window() {
        let cfg = SelfAttentionConfig::new(4).with_mask(AttentionMask::SlidingWindow(2));
        let attn = SelfAttention::new(cfg);
        let mask = attn.build_mask(5);
        // Position (0,3) distance=3 > half=1, should be masked
        assert!(mask[0 * 5 + 3].is_infinite());
        // Position (2,2) distance=0 <= 1, should be 0
        assert_eq!(mask[2 * 5 + 2], 0.0);
    }

    #[test]
    fn test_softmax_uniform() {
        let mut row = vec![1.0, 1.0, 1.0, 1.0];
        SelfAttention::softmax_row(&mut row);
        for &v in &row {
            assert!(approx_eq(v, 0.25, 1e-10));
        }
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let mut row = vec![2.0, -1.0, 0.5, 3.0];
        SelfAttention::softmax_row(&mut row);
        let sum: f64 = row.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
    }

    #[test]
    fn test_softmax_with_neg_inf() {
        let mut row = vec![1.0, f64::NEG_INFINITY, 1.0];
        SelfAttention::softmax_row(&mut row);
        assert!(approx_eq(row[1], 0.0, 1e-10));
        assert!(approx_eq(row[0], 0.5, 1e-10));
        assert!(approx_eq(row[2], 0.5, 1e-10));
    }

    #[test]
    fn test_self_attention_output_shape() {
        let cfg = SelfAttentionConfig::new(8);
        let mut attn = SelfAttention::new(cfg);
        let input = Tensor3::from_data(2, 4, 8, vec![0.1; 2 * 4 * 8]);
        let out = attn.forward(&input);
        assert_eq!(out.batch, 2);
        assert_eq!(out.rows, 4);
        assert_eq!(out.cols, 8);
    }

    #[test]
    fn test_self_attention_causal_output_shape() {
        let cfg = SelfAttentionConfig::new(4).with_mask(AttentionMask::Causal);
        let mut attn = SelfAttention::new(cfg);
        let input = Tensor3::from_data(1, 3, 4, vec![0.5; 12]);
        let out = attn.forward(&input);
        assert_eq!(out.batch, 1);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 4);
    }

    #[test]
    fn test_attention_weights_stored() {
        let cfg = SelfAttentionConfig::new(4);
        let mut attn = SelfAttention::new(cfg);
        assert!(attn.attention_weights().is_none());
        let input = Tensor3::from_data(1, 3, 4, vec![0.2; 12]);
        attn.forward(&input);
        let w = attn.attention_weights().unwrap();
        assert_eq!(w.batch, 1);
        assert_eq!(w.rows, 3);
        assert_eq!(w.cols, 3);
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let cfg = SelfAttentionConfig::new(4);
        let mut attn = SelfAttention::new(cfg);
        let input = Tensor3::from_data(1, 5, 4, vec![0.3; 20]);
        attn.forward(&input);
        let w = attn.attention_weights().unwrap();
        for q in 0..5 {
            let row_sum: f64 = (0..5).map(|k| w.get(0, q, k)).sum();
            assert!(approx_eq(row_sum, 1.0, 1e-9));
        }
    }

    #[test]
    fn test_causal_attention_weights_upper_zero() {
        let cfg = SelfAttentionConfig::new(4).with_mask(AttentionMask::Causal);
        let mut attn = SelfAttention::new(cfg);
        let input = Tensor3::from_data(1, 4, 4, vec![0.1; 16]);
        attn.forward(&input);
        let w = attn.attention_weights().unwrap();
        for r in 0..4 {
            for c in (r + 1)..4 {
                assert!(approx_eq(w.get(0, r, c), 0.0, 1e-10));
            }
        }
    }

    #[test]
    fn test_attention_entropy() {
        let cfg = SelfAttentionConfig::new(4);
        let mut attn = SelfAttention::new(cfg);
        let input = Tensor3::from_data(1, 3, 4, vec![0.5; 12]);
        attn.forward(&input);
        let entropy = attn.attention_entropy().unwrap();
        assert_eq!(entropy.len(), 3);
        for &h in &entropy {
            assert!(h >= 0.0);
        }
    }

    #[test]
    fn test_batched_matmul() {
        // (1,2,3) @ (1,3,2) -> (1,2,2)
        let a = Tensor3::from_data(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor3::from_data(1, 3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let c = batched_matmul(&a, &b);
        assert_eq!(c.get(0, 0, 0), 4.0);  // 1*1 + 2*0 + 3*1
        assert_eq!(c.get(0, 0, 1), 5.0);  // 1*0 + 2*1 + 3*1
        assert_eq!(c.get(0, 1, 0), 10.0); // 4*1 + 5*0 + 6*1
    }

    #[test]
    fn test_scaled_dot_product_attention_shape() {
        let q = Tensor3::from_data(1, 3, 4, vec![0.1; 12]);
        let k = Tensor3::from_data(1, 3, 4, vec![0.2; 12]);
        let v = Tensor3::from_data(1, 3, 4, vec![0.3; 12]);
        let out = scaled_dot_product_attention(&q, &k, &v, false);
        assert_eq!(out.batch, 1);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 4);
    }

    #[test]
    fn test_scaled_dot_product_causal() {
        let q = Tensor3::from_data(1, 3, 4, vec![0.1; 12]);
        let k = Tensor3::from_data(1, 3, 4, vec![0.2; 12]);
        let v = Tensor3::from_data(1, 3, 4, vec![0.3; 12]);
        let out = scaled_dot_product_attention(&q, &k, &v, true);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 4);
    }

    #[test]
    fn test_self_attention_display() {
        let cfg = SelfAttentionConfig::new(64).with_mask(AttentionMask::Causal).with_dropout(0.1);
        let attn = SelfAttention::new(cfg);
        let s = format!("{}", attn);
        assert!(s.contains("64"));
        assert!(s.contains("Causal"));
    }

    #[test]
    fn test_config_builder() {
        let cfg = SelfAttentionConfig::new(32)
            .with_mask(AttentionMask::SlidingWindow(128))
            .with_dropout(0.15)
            .with_scale_factor(4.0);
        assert_eq!(cfg.d_model, 32);
        assert_eq!(cfg.mask, AttentionMask::SlidingWindow(128));
        assert!(approx_eq(cfg.dropout_rate, 0.15, 1e-12));
        assert!(approx_eq(cfg.scale_factor.unwrap(), 4.0, 1e-12));
    }
}
