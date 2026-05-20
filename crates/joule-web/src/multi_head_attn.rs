//! Multi-head attention with parallel heads, concatenation, and output projection.
//!
//! Splits Q, K, V into `num_heads` parallel attention heads, computes scaled
//! dot-product attention independently per head, concatenates the results,
//! and applies a final linear projection. Supports both self-attention and
//! cross-attention (encoder-decoder) usage patterns.

use std::fmt;

// ── Dense Matrix ─────────────────────────────────────────────────

/// A 2-D dense matrix (row-major).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Matrix data length mismatch");
        Self { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Matrix multiply: (M, K) @ (K, N) -> (M, N).
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "inner dims must match");
        let mut out = Matrix::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut acc = 0.0;
                for k in 0..self.cols {
                    acc += self.get(r, k) * other.get(k, c);
                }
                out.set(r, c, acc);
            }
        }
        out
    }

    /// Transpose: (M, N) -> (N, M).
    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix({}x{})", self.rows, self.cols)
    }
}

// ── Linear Layer ─────────────────────────────────────────────────

/// Dense linear layer: y = x @ W^T + b.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Matrix,
    pub bias: Vec<f64>,
}

impl Linear {
    /// Initialize with deterministic pseudo-random weights.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut data = Vec::with_capacity(out_features * in_features);
        let mut state = seed;
        for _ in 0..(out_features * in_features) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            data.push(u * 2.0 * limit - limit);
        }
        Self {
            weight: Matrix::from_data(out_features, in_features, data),
            bias: vec![0.0; out_features],
        }
    }

    /// Forward: (S, in_features) -> (S, out_features).
    pub fn forward(&self, input: &Matrix) -> Matrix {
        assert_eq!(input.cols, self.weight.cols, "input cols != in_features");
        let wt = self.weight.transpose();
        let mut out = input.matmul(&wt);
        for r in 0..out.rows {
            for c in 0..out.cols {
                let v = out.get(r, c) + self.bias[c];
                out.set(r, c, v);
            }
        }
        out
    }
}

impl fmt::Display for Linear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear({} -> {})", self.weight.cols, self.weight.rows)
    }
}

// ── Softmax Utility ──────────────────────────────────────────────

/// In-place softmax over a mutable slice.
fn softmax_inplace(row: &mut [f64]) {
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

// ── Multi-Head Attention Config ──────────────────────────────────

/// Configuration for multi-head attention.
#[derive(Debug, Clone)]
pub struct MultiHeadAttnConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub causal: bool,
    pub dropout_rate: f64,
}

impl MultiHeadAttnConfig {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        Self { d_model, num_heads, causal: false, dropout_rate: 0.0 }
    }

    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    /// Dimension per head.
    pub fn d_k(&self) -> usize {
        self.d_model / self.num_heads
    }
}

// ── Multi-Head Attention ─────────────────────────────────────────

/// Multi-head attention layer.
///
/// Splits input into `num_heads` parallel attention heads of dimension
/// `d_k = d_model / num_heads`, applies scaled dot-product attention
/// per head, concatenates, and projects back to d_model.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub config: MultiHeadAttnConfig,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    last_attn_weights: Vec<Matrix>,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer.
    pub fn new(config: MultiHeadAttnConfig) -> Self {
        let d = config.d_model;
        Self {
            w_q: Linear::new(d, d, 100),
            w_k: Linear::new(d, d, 200),
            w_v: Linear::new(d, d, 300),
            w_o: Linear::new(d, d, 400),
            config,
            last_attn_weights: Vec::new(),
        }
    }

    /// Split a projected matrix (S, d_model) into num_heads matrices of (S, d_k).
    fn split_heads(&self, projected: &Matrix) -> Vec<Matrix> {
        let d_k = self.config.d_k();
        let h = self.config.num_heads;
        let s = projected.rows;
        let mut heads = Vec::with_capacity(h);
        for hi in 0..h {
            let mut head = Matrix::zeros(s, d_k);
            for r in 0..s {
                for c in 0..d_k {
                    head.set(r, c, projected.get(r, hi * d_k + c));
                }
            }
            heads.push(head);
        }
        heads
    }

    /// Concatenate per-head outputs (each S x d_k) into (S, d_model).
    fn concat_heads(&self, heads: &[Matrix]) -> Matrix {
        let s = heads[0].rows;
        let d_k = self.config.d_k();
        let d_model = self.config.d_model;
        let mut out = Matrix::zeros(s, d_model);
        for (hi, head) in heads.iter().enumerate() {
            for r in 0..s {
                for c in 0..d_k {
                    out.set(r, hi * d_k + c, head.get(r, c));
                }
            }
        }
        out
    }

    /// Single-head scaled dot-product attention.
    ///
    /// q, k, v: (S_q, d_k), (S_k, d_k), (S_k, d_k) -> (S_q, d_k)
    fn single_head_attention(&self, q: &Matrix, k: &Matrix, v: &Matrix) -> (Matrix, Matrix) {
        let s_q = q.rows;
        let s_k = k.rows;
        let d_k = q.cols;
        let scale = (d_k as f64).sqrt();

        // Scores: Q @ K^T / sqrt(d_k)
        let kt = k.transpose();
        let mut scores = q.matmul(&kt);
        for val in scores.data.iter_mut() {
            *val /= scale;
        }

        // Causal mask
        if self.config.causal {
            for r in 0..s_q {
                for c in (r + 1)..s_k {
                    scores.set(r, c, f64::NEG_INFINITY);
                }
            }
        }

        // Softmax per query
        for r in 0..s_q {
            let start = r * s_k;
            softmax_inplace(&mut scores.data[start..start + s_k]);
        }

        let attn_weights = scores.clone();
        let output = scores.matmul(v);
        (output, attn_weights)
    }

    /// Self-attention forward pass: query = key = value = input.
    ///
    /// input: (S, d_model) -> output: (S, d_model)
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        self.forward_cross(input, input, input)
    }

    /// Cross-attention forward pass with separate query and key/value sources.
    ///
    /// query: (S_q, d_model), key: (S_k, d_model), value: (S_k, d_model)
    /// -> output: (S_q, d_model)
    pub fn forward_cross(&mut self, query: &Matrix, key: &Matrix, value: &Matrix) -> Matrix {
        let q_proj = self.w_q.forward(query);
        let k_proj = self.w_k.forward(key);
        let v_proj = self.w_v.forward(value);

        let q_heads = self.split_heads(&q_proj);
        let k_heads = self.split_heads(&k_proj);
        let v_heads = self.split_heads(&v_proj);

        let mut head_outputs = Vec::with_capacity(self.config.num_heads);
        self.last_attn_weights.clear();

        for hi in 0..self.config.num_heads {
            let (out, weights) = self.single_head_attention(&q_heads[hi], &k_heads[hi], &v_heads[hi]);
            head_outputs.push(out);
            self.last_attn_weights.push(weights);
        }

        let concatenated = self.concat_heads(&head_outputs);
        self.w_o.forward(&concatenated)
    }

    /// Access attention weights from the last forward pass, per head.
    pub fn attention_weights(&self) -> &[Matrix] {
        &self.last_attn_weights
    }

    /// Average attention weights across all heads.
    pub fn average_attention(&self) -> Option<Matrix> {
        if self.last_attn_weights.is_empty() {
            return None;
        }
        let s_q = self.last_attn_weights[0].rows;
        let s_k = self.last_attn_weights[0].cols;
        let h = self.last_attn_weights.len() as f64;
        let mut avg = Matrix::zeros(s_q, s_k);
        for w in &self.last_attn_weights {
            for r in 0..s_q {
                for c in 0..s_k {
                    let v = avg.get(r, c) + w.get(r, c) / h;
                    avg.set(r, c, v);
                }
            }
        }
        Some(avg)
    }

    /// Total number of parameters in the layer.
    pub fn num_parameters(&self) -> usize {
        let linear_params = |l: &Linear| l.weight.rows * l.weight.cols + l.bias.len();
        linear_params(&self.w_q) + linear_params(&self.w_k)
            + linear_params(&self.w_v) + linear_params(&self.w_o)
    }
}

impl fmt::Display for MultiHeadAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiHeadAttention(d_model={}, heads={}, d_k={}, causal={})",
            self.config.d_model,
            self.config.num_heads,
            self.config.d_k(),
            self.config.causal,
        )
    }
}

// ── Attention Head Statistics ────────────────────────────────────

/// Summary statistics for a single attention head.
#[derive(Debug, Clone)]
pub struct HeadStats {
    pub head_index: usize,
    pub mean_entropy: f64,
    pub max_weight: f64,
    pub sparsity: f64,
}

impl fmt::Display for HeadStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Head[{}]: entropy={:.4}, max={:.4}, sparsity={:.4}",
            self.head_index, self.mean_entropy, self.max_weight, self.sparsity
        )
    }
}

/// Compute per-head statistics from attention weights.
pub fn compute_head_stats(weights: &[Matrix]) -> Vec<HeadStats> {
    weights.iter().enumerate().map(|(hi, w)| {
        let s_q = w.rows;
        let s_k = w.cols;
        let mut total_entropy = 0.0;
        let mut max_w = 0.0_f64;
        let mut near_zero = 0usize;
        let threshold = 1e-4;

        for r in 0..s_q {
            let mut row_entropy = 0.0;
            for c in 0..s_k {
                let p = w.get(r, c);
                max_w = max_w.max(p);
                if p > 1e-12 {
                    row_entropy -= p * p.ln();
                }
                if p < threshold {
                    near_zero += 1;
                }
            }
            total_entropy += row_entropy;
        }

        HeadStats {
            head_index: hi,
            mean_entropy: total_entropy / s_q as f64,
            max_weight: max_w,
            sparsity: near_zero as f64 / (s_q * s_k) as f64,
        }
    }).collect()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_matrix_zeros() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.data.len(), 12);
        assert!(m.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_matrix_matmul() {
        let a = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_data(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let c = a.matmul(&b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.get(0, 0), 4.0);
        assert_eq!(c.get(0, 1), 5.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_matrix_display() {
        let m = Matrix::zeros(5, 8);
        assert_eq!(format!("{}", m), "Matrix(5x8)");
    }

    #[test]
    fn test_linear_forward_shape() {
        let lin = Linear::new(8, 16, 42);
        let input = Matrix::zeros(5, 8);
        let out = lin.forward(&input);
        assert_eq!(out.rows, 5);
        assert_eq!(out.cols, 16);
    }

    #[test]
    fn test_softmax_inplace() {
        let mut row = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut row);
        let sum: f64 = row.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
        assert!(row[2] > row[1]);
        assert!(row[1] > row[0]);
    }

    #[test]
    fn test_config_d_k() {
        let cfg = MultiHeadAttnConfig::new(64, 8);
        assert_eq!(cfg.d_k(), 8);
    }

    #[test]
    fn test_config_builder() {
        let cfg = MultiHeadAttnConfig::new(128, 4)
            .with_causal(true)
            .with_dropout(0.1);
        assert!(cfg.causal);
        assert!(approx_eq(cfg.dropout_rate, 0.1, 1e-12));
    }

    #[test]
    fn test_mha_output_shape() {
        let cfg = MultiHeadAttnConfig::new(16, 4);
        let mut mha = MultiHeadAttention::new(cfg);
        let input = Matrix::from_data(6, 16, vec![0.1; 96]);
        let out = mha.forward(&input);
        assert_eq!(out.rows, 6);
        assert_eq!(out.cols, 16);
    }

    #[test]
    fn test_mha_causal_output_shape() {
        let cfg = MultiHeadAttnConfig::new(8, 2).with_causal(true);
        let mut mha = MultiHeadAttention::new(cfg);
        let input = Matrix::from_data(4, 8, vec![0.2; 32]);
        let out = mha.forward(&input);
        assert_eq!(out.rows, 4);
        assert_eq!(out.cols, 8);
    }

    #[test]
    fn test_mha_cross_attention() {
        let cfg = MultiHeadAttnConfig::new(8, 2);
        let mut mha = MultiHeadAttention::new(cfg);
        let q = Matrix::from_data(3, 8, vec![0.1; 24]);
        let kv = Matrix::from_data(5, 8, vec![0.2; 40]);
        let out = mha.forward_cross(&q, &kv, &kv);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 8);
    }

    #[test]
    fn test_attention_weights_per_head() {
        let cfg = MultiHeadAttnConfig::new(8, 2);
        let mut mha = MultiHeadAttention::new(cfg);
        let input = Matrix::from_data(4, 8, vec![0.3; 32]);
        mha.forward(&input);
        let weights = mha.attention_weights();
        assert_eq!(weights.len(), 2);
        for w in weights {
            assert_eq!(w.rows, 4);
            assert_eq!(w.cols, 4);
        }
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let cfg = MultiHeadAttnConfig::new(8, 2);
        let mut mha = MultiHeadAttention::new(cfg);
        let input = Matrix::from_data(3, 8, vec![0.5; 24]);
        mha.forward(&input);
        for w in mha.attention_weights() {
            for r in 0..w.rows {
                let sum: f64 = (0..w.cols).map(|c| w.get(r, c)).sum();
                assert!(approx_eq(sum, 1.0, 1e-9));
            }
        }
    }

    #[test]
    fn test_average_attention() {
        let cfg = MultiHeadAttnConfig::new(8, 2);
        let mut mha = MultiHeadAttention::new(cfg);
        let input = Matrix::from_data(3, 8, vec![0.1; 24]);
        mha.forward(&input);
        let avg = mha.average_attention().unwrap();
        assert_eq!(avg.rows, 3);
        assert_eq!(avg.cols, 3);
    }

    #[test]
    fn test_num_parameters() {
        let cfg = MultiHeadAttnConfig::new(16, 4);
        let mha = MultiHeadAttention::new(cfg);
        // 4 linear layers, each 16*16 + 16 = 272 params
        assert_eq!(mha.num_parameters(), 4 * (16 * 16 + 16));
    }

    #[test]
    fn test_split_concat_roundtrip() {
        let cfg = MultiHeadAttnConfig::new(8, 2);
        let mha = MultiHeadAttention::new(cfg);
        let m = Matrix::from_data(3, 8, (0..24).map(|i| i as f64).collect());
        let heads = mha.split_heads(&m);
        let reconstructed = mha.concat_heads(&heads);
        assert_eq!(m.data, reconstructed.data);
    }

    #[test]
    fn test_head_stats() {
        let cfg = MultiHeadAttnConfig::new(8, 2);
        let mut mha = MultiHeadAttention::new(cfg);
        let input = Matrix::from_data(4, 8, vec![0.2; 32]);
        mha.forward(&input);
        let stats = compute_head_stats(mha.attention_weights());
        assert_eq!(stats.len(), 2);
        for s in &stats {
            assert!(s.mean_entropy >= 0.0);
            assert!(s.max_weight >= 0.0);
            assert!(s.sparsity >= 0.0 && s.sparsity <= 1.0);
        }
    }

    #[test]
    fn test_head_stats_display() {
        let s = HeadStats { head_index: 0, mean_entropy: 1.5, max_weight: 0.8, sparsity: 0.3 };
        let disp = format!("{}", s);
        assert!(disp.contains("Head[0]"));
        assert!(disp.contains("entropy"));
    }

    #[test]
    fn test_mha_display() {
        let cfg = MultiHeadAttnConfig::new(32, 4).with_causal(true);
        let mha = MultiHeadAttention::new(cfg);
        let s = format!("{}", mha);
        assert!(s.contains("32"));
        assert!(s.contains("4"));
        assert!(s.contains("causal=true"));
    }
}
