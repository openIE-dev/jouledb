//! Transformer decoder with masked self-attention, cross-attention, and autoregressive generation.
//!
//! Implements the decoder side of the Transformer architecture with causal
//! (masked) self-attention to prevent attending to future positions,
//! cross-attention over encoder outputs, position-wise feed-forward blocks,
//! and support for greedy autoregressive token generation.

use std::fmt;

// ── Dense Matrix ─────────────────────────────────────────────────

/// Row-major dense matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct DMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "DMatrix data length mismatch");
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

    pub fn transpose(&self) -> DMatrix {
        let mut out = DMatrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    pub fn matmul(&self, other: &DMatrix) -> DMatrix {
        assert_eq!(self.cols, other.rows, "inner dims must match");
        let mut out = DMatrix::zeros(self.rows, other.cols);
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

    pub fn add(&self, other: &DMatrix) -> DMatrix {
        assert_eq!(self.data.len(), other.data.len());
        let data: Vec<f64> = self.data.iter().zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        DMatrix { rows: self.rows, cols: self.cols, data }
    }

    /// Extract a single row as a new (1, cols) matrix.
    pub fn row(&self, r: usize) -> DMatrix {
        let start = r * self.cols;
        DMatrix::from_data(1, self.cols, self.data[start..start + self.cols].to_vec())
    }

    /// Append a row to this matrix, creating (rows+1, cols).
    pub fn append_row(&self, row: &DMatrix) -> DMatrix {
        assert_eq!(row.cols, self.cols);
        assert_eq!(row.rows, 1);
        let mut data = self.data.clone();
        data.extend_from_slice(&row.data);
        DMatrix { rows: self.rows + 1, cols: self.cols, data }
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

impl fmt::Display for DMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DMatrix({}x{})", self.rows, self.cols)
    }
}

// ── Internal Components ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct Linear {
    weight: DMatrix,
    bias: Vec<f64>,
}

impl Linear {
    fn new(in_f: usize, out_f: usize, seed: u64) -> Self {
        let limit = (6.0 / (in_f + out_f) as f64).sqrt();
        let mut data = Vec::with_capacity(out_f * in_f);
        let mut state = seed;
        for _ in 0..(out_f * in_f) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64;
            data.push(u * 2.0 * limit - limit);
        }
        Self { weight: DMatrix::from_data(out_f, in_f, data), bias: vec![0.0; out_f] }
    }

    fn forward(&self, input: &DMatrix) -> DMatrix {
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

#[derive(Debug, Clone)]
struct LNorm {
    gamma: Vec<f64>,
    beta: Vec<f64>,
    eps: f64,
    dim: usize,
}

impl LNorm {
    fn new(dim: usize) -> Self {
        Self { gamma: vec![1.0; dim], beta: vec![0.0; dim], eps: 1e-5, dim }
    }

    fn forward(&self, input: &DMatrix) -> DMatrix {
        let mut out = DMatrix::zeros(input.rows, input.cols);
        for r in 0..input.rows {
            let mut mean = 0.0;
            for c in 0..self.dim {
                mean += input.get(r, c);
            }
            mean /= self.dim as f64;
            let mut var = 0.0;
            for c in 0..self.dim {
                let diff = input.get(r, c) - mean;
                var += diff * diff;
            }
            var /= self.dim as f64;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for c in 0..self.dim {
                let normed = (input.get(r, c) - mean) * inv_std;
                out.set(r, c, self.gamma[c] * normed + self.beta[c]);
            }
        }
        out
    }
}

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

fn gelu(x: f64) -> f64 {
    x * 0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    sign * (1.0 - poly * (-a * a).exp())
}

// ── Masked Multi-Head Attention ──────────────────────────────────

#[derive(Debug, Clone)]
struct MaskedMHAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    num_heads: usize,
    d_k: usize,
    d_model: usize,
    causal: bool,
}

impl MaskedMHAttention {
    fn new(d_model: usize, num_heads: usize, causal: bool, seed: u64) -> Self {
        assert_eq!(d_model % num_heads, 0);
        Self {
            w_q: Linear::new(d_model, d_model, seed),
            w_k: Linear::new(d_model, d_model, seed + 1000),
            w_v: Linear::new(d_model, d_model, seed + 2000),
            w_o: Linear::new(d_model, d_model, seed + 3000),
            num_heads,
            d_k: d_model / num_heads,
            d_model,
            causal,
        }
    }

    fn forward(&self, query: &DMatrix, key: &DMatrix, value: &DMatrix) -> DMatrix {
        let s_q = query.rows;
        let s_k = key.rows;
        let q = self.w_q.forward(query);
        let k = self.w_k.forward(key);
        let v = self.w_v.forward(value);
        let scale = (self.d_k as f64).sqrt();

        let mut head_outputs = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let offset = h * self.d_k;
            let mut qh = DMatrix::zeros(s_q, self.d_k);
            let mut kh = DMatrix::zeros(s_k, self.d_k);
            let mut vh = DMatrix::zeros(s_k, self.d_k);
            for r in 0..s_q {
                for c in 0..self.d_k {
                    qh.set(r, c, q.get(r, offset + c));
                }
            }
            for r in 0..s_k {
                for c in 0..self.d_k {
                    kh.set(r, c, k.get(r, offset + c));
                    vh.set(r, c, v.get(r, offset + c));
                }
            }
            let kt = kh.transpose();
            let mut scores = qh.matmul(&kt);
            for val in scores.data.iter_mut() {
                *val /= scale;
            }
            if self.causal {
                for r in 0..s_q {
                    for c in (r + 1)..s_k {
                        scores.set(r, c, f64::NEG_INFINITY);
                    }
                }
            }
            for r in 0..s_q {
                let start = r * s_k;
                softmax_row(&mut scores.data[start..start + s_k]);
            }
            head_outputs.push(scores.matmul(&vh));
        }

        let mut concat = DMatrix::zeros(s_q, self.d_model);
        for (h, ho) in head_outputs.iter().enumerate() {
            let offset = h * self.d_k;
            for r in 0..s_q {
                for c in 0..self.d_k {
                    concat.set(r, offset + c, ho.get(r, c));
                }
            }
        }
        self.w_o.forward(&concat)
    }
}

// ── Feed-Forward ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct FFBlock {
    linear1: Linear,
    linear2: Linear,
}

impl FFBlock {
    fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            linear1: Linear::new(d_model, d_ff, seed),
            linear2: Linear::new(d_ff, d_model, seed + 500),
        }
    }

    fn forward(&self, input: &DMatrix) -> DMatrix {
        let mut hidden = self.linear1.forward(input);
        for val in hidden.data.iter_mut() {
            *val = gelu(*val);
        }
        self.linear2.forward(&hidden)
    }
}

// ── Decoder Layer ────────────────────────────────────────────────

/// A single transformer decoder layer with:
/// 1. Masked (causal) multi-head self-attention
/// 2. Cross-attention over encoder output
/// 3. Position-wise feed-forward network
#[derive(Debug, Clone)]
pub struct DecoderLayer {
    self_attn: MaskedMHAttention,
    cross_attn: MaskedMHAttention,
    ffn: FFBlock,
    norm1: LNorm,
    norm2: LNorm,
    norm3: LNorm,
    pre_norm: bool,
}

impl DecoderLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize, pre_norm: bool, seed: u64) -> Self {
        Self {
            self_attn: MaskedMHAttention::new(d_model, num_heads, true, seed),
            cross_attn: MaskedMHAttention::new(d_model, num_heads, false, seed + 5000),
            ffn: FFBlock::new(d_model, d_ff, seed + 10000),
            norm1: LNorm::new(d_model),
            norm2: LNorm::new(d_model),
            norm3: LNorm::new(d_model),
            pre_norm,
        }
    }

    fn forward(&self, target: &DMatrix, encoder_out: &DMatrix) -> DMatrix {
        // Sub-layer 1: masked self-attention
        let sa_out = if self.pre_norm {
            let normed = self.norm1.forward(target);
            let attn = self.self_attn.forward(&normed, &normed, &normed);
            target.add(&attn)
        } else {
            let attn = self.self_attn.forward(target, target, target);
            let residual = target.add(&attn);
            self.norm1.forward(&residual)
        };

        // Sub-layer 2: cross-attention
        let ca_out = if self.pre_norm {
            let normed = self.norm2.forward(&sa_out);
            let attn = self.cross_attn.forward(&normed, encoder_out, encoder_out);
            sa_out.add(&attn)
        } else {
            let attn = self.cross_attn.forward(&sa_out, encoder_out, encoder_out);
            let residual = sa_out.add(&attn);
            self.norm2.forward(&residual)
        };

        // Sub-layer 3: feed-forward
        if self.pre_norm {
            let normed = self.norm3.forward(&ca_out);
            let ff = self.ffn.forward(&normed);
            ca_out.add(&ff)
        } else {
            let ff = self.ffn.forward(&ca_out);
            let residual = ca_out.add(&ff);
            self.norm3.forward(&residual)
        }
    }
}

// ── Decoder Configuration ────────────────────────────────────────

/// Configuration for the transformer decoder.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub pre_norm: bool,
    pub dropout_rate: f64,
}

impl DecoderConfig {
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize, vocab_size: usize) -> Self {
        Self {
            d_model,
            num_heads,
            num_layers,
            d_ff: d_model * 4,
            vocab_size,
            max_seq_len: 512,
            pre_norm: true,
            dropout_rate: 0.0,
        }
    }

    pub fn with_d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }

    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }
}

// ── Transformer Decoder ──────────────────────────────────────────

/// Full transformer decoder: N stacked decoder layers with an output
/// projection to vocabulary logits.
///
/// Supports both teacher-forced forward passes (given full target sequence)
/// and autoregressive generation (one token at a time).
#[derive(Debug, Clone)]
pub struct TransformerDecoder {
    pub config: DecoderConfig,
    layers: Vec<DecoderLayer>,
    final_norm: Option<LNorm>,
    output_proj: Linear,
}

impl TransformerDecoder {
    /// Create a new transformer decoder.
    pub fn new(config: DecoderConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let seed = (i as u64 + 1) * 8191;
            layers.push(DecoderLayer::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.pre_norm,
                seed,
            ));
        }
        let final_norm = if config.pre_norm {
            Some(LNorm::new(config.d_model))
        } else {
            None
        };
        let output_proj = Linear::new(config.d_model, config.vocab_size, 99999);
        Self { config, layers, final_norm, output_proj }
    }

    /// Forward pass with teacher forcing.
    ///
    /// target: (target_len, d_model), encoder_out: (src_len, d_model)
    /// Returns logits: (target_len, vocab_size).
    pub fn forward(&self, target: &DMatrix, encoder_out: &DMatrix) -> DMatrix {
        assert_eq!(target.cols, self.config.d_model);
        assert_eq!(encoder_out.cols, self.config.d_model);

        let mut hidden = target.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, encoder_out);
        }
        if let Some(ref norm) = self.final_norm {
            hidden = norm.forward(&hidden);
        }
        self.output_proj.forward(&hidden)
    }

    /// Greedy autoregressive generation.
    ///
    /// Starts from an initial token embedding and generates up to `max_steps`
    /// tokens by always selecting the argmax of the output logits.
    pub fn generate_greedy(
        &self,
        initial: &DMatrix,
        encoder_out: &DMatrix,
        max_steps: usize,
    ) -> Vec<usize> {
        let mut generated = Vec::new();
        let mut sequence = initial.clone();

        for _ in 0..max_steps {
            let logits = self.forward(&sequence, encoder_out);
            // Take logits from the last position
            let last_row = logits.rows - 1;
            let mut best_token = 0;
            let mut best_score = f64::NEG_INFINITY;
            for v in 0..self.config.vocab_size {
                let score = logits.get(last_row, v);
                if score > best_score {
                    best_score = score;
                    best_token = v;
                }
            }
            generated.push(best_token);

            // Create next step embedding (simplified: one-hot scaled)
            let mut next_emb = DMatrix::zeros(1, self.config.d_model);
            if best_token < self.config.d_model {
                next_emb.set(0, best_token, 1.0);
            }
            sequence = sequence.append_row(&next_emb);

            if sequence.rows >= self.config.max_seq_len {
                break;
            }
        }
        generated
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total parameter count.
    pub fn num_parameters(&self) -> usize {
        let d = self.config.d_model;
        let d_ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let n = self.config.num_layers;
        // Per layer: 2 MHA (4 linears each) + 1 FFN (2 linears) + 3 norms
        let mha_params = 4 * (d * d + d);
        let ffn_params = (d * d_ff + d_ff) + (d_ff * d + d);
        let norm_params = 3 * (2 * d);
        let per_layer = 2 * mha_params + ffn_params + norm_params;
        let final_norm_params = if self.final_norm.is_some() { 2 * d } else { 0 };
        let output_params = d * v + v;
        n * per_layer + final_norm_params + output_params
    }
}

impl fmt::Display for TransformerDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformerDecoder(layers={}, d_model={}, heads={}, vocab={}, pre_norm={})",
            self.config.num_layers,
            self.config.d_model,
            self.config.num_heads,
            self.config.vocab_size,
            self.config.pre_norm,
        )
    }
}

// ── Temperature Sampling ─────────────────────────────────────────

/// Apply temperature scaling to logits before sampling.
pub fn apply_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    assert!(temperature > 0.0, "temperature must be positive");
    logits.iter().map(|l| l / temperature).collect()
}

/// Convert logits to probabilities via softmax.
pub fn logits_to_probs(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Top-k filtering: set all logits outside top-k to negative infinity.
pub fn top_k_filter(logits: &[f64], k: usize) -> Vec<f64> {
    if k >= logits.len() {
        return logits.to_vec();
    }
    let mut indexed: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = indexed[k - 1].1;
    logits.iter().map(|v| {
        if *v >= threshold { *v } else { f64::NEG_INFINITY }
    }).collect()
}

/// Top-p (nucleus) filtering: keep smallest set of tokens whose cumulative
/// probability exceeds p.
pub fn top_p_filter(logits: &[f64], p: f64) -> Vec<f64> {
    let probs = logits_to_probs(logits);
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut keep = vec![false; logits.len()];
    for &(idx, prob) in &indexed {
        keep[idx] = true;
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }

    logits.iter().enumerate().map(|(i, &v)| {
        if keep[i] { v } else { f64::NEG_INFINITY }
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
    fn test_dmatrix_zeros() {
        let m = DMatrix::zeros(3, 4);
        assert_eq!(m.data.len(), 12);
        assert!(m.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_dmatrix_row() {
        let m = DMatrix::from_data(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = m.row(1);
        assert_eq!(r.rows, 1);
        assert_eq!(r.cols, 2);
        assert_eq!(r.get(0, 0), 3.0);
        assert_eq!(r.get(0, 1), 4.0);
    }

    #[test]
    fn test_dmatrix_append_row() {
        let m = DMatrix::from_data(2, 3, vec![1.0; 6]);
        let row = DMatrix::from_data(1, 3, vec![2.0; 3]);
        let appended = m.append_row(&row);
        assert_eq!(appended.rows, 3);
        assert_eq!(appended.cols, 3);
        assert_eq!(appended.get(2, 0), 2.0);
    }

    #[test]
    fn test_dmatrix_display() {
        let m = DMatrix::zeros(5, 8);
        assert_eq!(format!("{}", m), "DMatrix(5x8)");
    }

    #[test]
    fn test_decoder_config_defaults() {
        let cfg = DecoderConfig::new(64, 8, 6, 1000);
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_layers, 6);
        assert_eq!(cfg.vocab_size, 1000);
        assert_eq!(cfg.d_ff, 256);
        assert!(cfg.pre_norm);
    }

    #[test]
    fn test_decoder_config_builder() {
        let cfg = DecoderConfig::new(32, 4, 3, 500)
            .with_d_ff(128)
            .with_max_seq_len(1024)
            .with_pre_norm(false)
            .with_dropout(0.1);
        assert_eq!(cfg.d_ff, 128);
        assert_eq!(cfg.max_seq_len, 1024);
        assert!(!cfg.pre_norm);
    }

    #[test]
    fn test_decoder_forward_shape() {
        let cfg = DecoderConfig::new(8, 2, 1, 16);
        let dec = TransformerDecoder::new(cfg);
        let target = DMatrix::from_data(3, 8, vec![0.1; 24]);
        let enc_out = DMatrix::from_data(5, 8, vec![0.2; 40]);
        let logits = dec.forward(&target, &enc_out);
        assert_eq!(logits.rows, 3);
        assert_eq!(logits.cols, 16);
    }

    #[test]
    fn test_decoder_two_layers() {
        let cfg = DecoderConfig::new(8, 2, 2, 10);
        let dec = TransformerDecoder::new(cfg);
        let target = DMatrix::from_data(4, 8, vec![0.1; 32]);
        let enc_out = DMatrix::from_data(3, 8, vec![0.2; 24]);
        let logits = dec.forward(&target, &enc_out);
        assert_eq!(logits.rows, 4);
        assert_eq!(logits.cols, 10);
    }

    #[test]
    fn test_decoder_post_norm() {
        let cfg = DecoderConfig::new(8, 2, 1, 10).with_pre_norm(false);
        let dec = TransformerDecoder::new(cfg);
        let target = DMatrix::from_data(3, 8, vec![0.1; 24]);
        let enc_out = DMatrix::from_data(2, 8, vec![0.2; 16]);
        let logits = dec.forward(&target, &enc_out);
        assert_eq!(logits.rows, 3);
    }

    #[test]
    fn test_greedy_generation() {
        let cfg = DecoderConfig::new(8, 2, 1, 16).with_max_seq_len(10);
        let dec = TransformerDecoder::new(cfg);
        let initial = DMatrix::from_data(1, 8, vec![0.1; 8]);
        let enc_out = DMatrix::from_data(3, 8, vec![0.2; 24]);
        let tokens = dec.generate_greedy(&initial, &enc_out, 5);
        assert_eq!(tokens.len(), 5);
        for &t in &tokens {
            assert!(t < 16);
        }
    }

    #[test]
    fn test_greedy_generation_max_len() {
        let cfg = DecoderConfig::new(8, 2, 1, 10).with_max_seq_len(4);
        let dec = TransformerDecoder::new(cfg);
        let initial = DMatrix::from_data(1, 8, vec![0.1; 8]);
        let enc_out = DMatrix::from_data(2, 8, vec![0.2; 16]);
        let tokens = dec.generate_greedy(&initial, &enc_out, 100);
        // Should stop at max_seq_len
        assert!(tokens.len() <= 4);
    }

    #[test]
    fn test_decoder_num_layers() {
        let cfg = DecoderConfig::new(16, 4, 6, 100);
        let dec = TransformerDecoder::new(cfg);
        assert_eq!(dec.num_layers(), 6);
    }

    #[test]
    fn test_decoder_num_parameters() {
        let cfg = DecoderConfig::new(8, 2, 1, 16);
        let dec = TransformerDecoder::new(cfg);
        let params = dec.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_decoder_display() {
        let cfg = DecoderConfig::new(64, 8, 6, 32000);
        let dec = TransformerDecoder::new(cfg);
        let s = format!("{}", dec);
        assert!(s.contains("layers=6"));
        assert!(s.contains("d_model=64"));
        assert!(s.contains("vocab=32000"));
    }

    #[test]
    fn test_temperature_scaling() {
        let logits = vec![1.0, 2.0, 3.0];
        let scaled = apply_temperature(&logits, 2.0);
        assert!(approx_eq(scaled[0], 0.5, 1e-10));
        assert!(approx_eq(scaled[1], 1.0, 1e-10));
    }

    #[test]
    fn test_logits_to_probs_sum_one() {
        let logits = vec![1.0, 2.0, 3.0, -1.0];
        let probs = logits_to_probs(&logits);
        let sum: f64 = probs.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
    }

    #[test]
    fn test_logits_to_probs_order() {
        let logits = vec![1.0, 3.0, 2.0];
        let probs = logits_to_probs(&logits);
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn test_top_k_filter() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let filtered = top_k_filter(&logits, 2);
        // Only top 2 (indices 1=5.0, 4=4.0) should remain
        assert!(filtered[0].is_infinite() && filtered[0] < 0.0);
        assert_eq!(filtered[1], 5.0);
        assert_eq!(filtered[4], 4.0);
    }

    #[test]
    fn test_top_k_filter_all() {
        let logits = vec![1.0, 2.0, 3.0];
        let filtered = top_k_filter(&logits, 10);
        assert_eq!(filtered, logits);
    }

    #[test]
    fn test_top_p_filter() {
        let logits = vec![10.0, 1.0, 0.0, -10.0];
        let filtered = top_p_filter(&logits, 0.95);
        // The highest logit (10.0) should dominate, most others masked
        assert_eq!(filtered[0], 10.0);
    }
}
