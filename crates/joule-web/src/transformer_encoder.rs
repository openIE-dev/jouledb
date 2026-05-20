//! Full transformer encoder stack with N layers, residual connections, and normalization.
//!
//! Implements the encoder side of the Transformer (Vaswani et al. 2017)
//! with configurable layer count, multi-head self-attention, position-wise
//! feed-forward blocks, pre-norm or post-norm layer normalization, and
//! residual connections. Supports attention weight extraction and per-layer
//! activation diagnostics.

use std::fmt;

// ── Dense Matrix ─────────────────────────────────────────────────

/// Row-major dense matrix for sequence data (seq_len x d_model).
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Mat data length mismatch");
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

    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "inner dims must match");
        let mut out = Mat::zeros(self.rows, other.cols);
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

    /// Element-wise addition.
    pub fn add(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Mat { rows: self.rows, cols: self.cols, data }
    }

    /// L2 norm of the entire matrix (Frobenius norm).
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mat({}x{})", self.rows, self.cols)
    }
}

// ── Linear Layer ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Linear {
    weight: Mat,
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
        Self { weight: Mat::from_data(out_f, in_f, data), bias: vec![0.0; out_f] }
    }

    fn forward(&self, input: &Mat) -> Mat {
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

// ── Layer Normalization ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct LayerNorm {
    gamma: Vec<f64>,
    beta: Vec<f64>,
    eps: f64,
    dim: usize,
}

impl LayerNorm {
    fn new(dim: usize) -> Self {
        Self { gamma: vec![1.0; dim], beta: vec![0.0; dim], eps: 1e-5, dim }
    }

    fn forward(&self, input: &Mat) -> Mat {
        let mut out = Mat::zeros(input.rows, input.cols);
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

// ── Softmax ──────────────────────────────────────────────────────

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

// ── GELU ─────────────────────────────────────────────────────────

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

// ── Multi-Head Attention ─────────────────────────────────────────

#[derive(Debug, Clone)]
struct MHAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    num_heads: usize,
    d_k: usize,
    d_model: usize,
    last_weights: Vec<Mat>,
}

impl MHAttention {
    fn new(d_model: usize, num_heads: usize, seed: u64) -> Self {
        assert_eq!(d_model % num_heads, 0);
        Self {
            w_q: Linear::new(d_model, d_model, seed),
            w_k: Linear::new(d_model, d_model, seed + 1000),
            w_v: Linear::new(d_model, d_model, seed + 2000),
            w_o: Linear::new(d_model, d_model, seed + 3000),
            num_heads,
            d_k: d_model / num_heads,
            d_model,
            last_weights: Vec::new(),
        }
    }

    fn forward(&mut self, input: &Mat) -> Mat {
        let s = input.rows;
        let q = self.w_q.forward(input);
        let k = self.w_k.forward(input);
        let v = self.w_v.forward(input);
        let scale = (self.d_k as f64).sqrt();

        let mut head_outputs = Vec::with_capacity(self.num_heads);
        self.last_weights.clear();

        for h in 0..self.num_heads {
            let offset = h * self.d_k;
            // Extract head slice
            let mut qh = Mat::zeros(s, self.d_k);
            let mut kh = Mat::zeros(s, self.d_k);
            let mut vh = Mat::zeros(s, self.d_k);
            for r in 0..s {
                for c in 0..self.d_k {
                    qh.set(r, c, q.get(r, offset + c));
                    kh.set(r, c, k.get(r, offset + c));
                    vh.set(r, c, v.get(r, offset + c));
                }
            }
            // Scores
            let kt = kh.transpose();
            let mut scores = qh.matmul(&kt);
            for val in scores.data.iter_mut() {
                *val /= scale;
            }
            for r in 0..s {
                let start = r * s;
                softmax_row(&mut scores.data[start..start + s]);
            }
            self.last_weights.push(scores.clone());
            let out = scores.matmul(&vh);
            head_outputs.push(out);
        }

        // Concat heads
        let mut concat = Mat::zeros(s, self.d_model);
        for (h, ho) in head_outputs.iter().enumerate() {
            let offset = h * self.d_k;
            for r in 0..s {
                for c in 0..self.d_k {
                    concat.set(r, offset + c, ho.get(r, c));
                }
            }
        }
        self.w_o.forward(&concat)
    }
}

// ── Feed-Forward Block ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    d_ff: usize,
}

impl FeedForward {
    fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            linear1: Linear::new(d_model, d_ff, seed),
            linear2: Linear::new(d_ff, d_model, seed + 500),
            d_ff,
        }
    }

    fn forward(&self, input: &Mat) -> Mat {
        let mut hidden = self.linear1.forward(input);
        for val in hidden.data.iter_mut() {
            *val = gelu(*val);
        }
        self.linear2.forward(&hidden)
    }
}

// ── Encoder Layer ────────────────────────────────────────────────

/// A single transformer encoder layer: self-attention + feed-forward
/// with residual connections and layer normalization.
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    self_attn: MHAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    pre_norm: bool,
}

impl EncoderLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize, pre_norm: bool, seed: u64) -> Self {
        Self {
            self_attn: MHAttention::new(d_model, num_heads, seed),
            ffn: FeedForward::new(d_model, d_ff, seed + 10000),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            pre_norm,
        }
    }

    fn forward(&mut self, input: &Mat) -> Mat {
        // Sub-layer 1: self-attention
        let attn_out = if self.pre_norm {
            let normed = self.norm1.forward(input);
            let attn = self.self_attn.forward(&normed);
            input.add(&attn)
        } else {
            let attn = self.self_attn.forward(input);
            let residual = input.add(&attn);
            self.norm1.forward(&residual)
        };

        // Sub-layer 2: feed-forward
        if self.pre_norm {
            let normed = self.norm2.forward(&attn_out);
            let ff = self.ffn.forward(&normed);
            attn_out.add(&ff)
        } else {
            let ff = self.ffn.forward(&attn_out);
            let residual = attn_out.add(&ff);
            self.norm2.forward(&residual)
        }
    }

    fn attention_weights(&self) -> &[Mat] {
        &self.self_attn.last_weights
    }
}

// ── Encoder Configuration ────────────────────────────────────────

/// Configuration for the full transformer encoder stack.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub dropout_rate: f64,
    pub pre_norm: bool,
}

impl EncoderConfig {
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize) -> Self {
        Self {
            d_model,
            num_heads,
            num_layers,
            d_ff: d_model * 4,
            dropout_rate: 0.0,
            pre_norm: true,
        }
    }

    pub fn with_d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }
}

// ── Transformer Encoder ──────────────────────────────────────────

/// Full transformer encoder: N stacked encoder layers with optional
/// final layer normalization.
///
/// Each layer consists of multi-head self-attention followed by a
/// position-wise feed-forward network, with residual connections
/// and layer normalization applied in either pre-norm or post-norm order.
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    pub config: EncoderConfig,
    layers: Vec<EncoderLayer>,
    final_norm: Option<LayerNorm>,
    /// Per-layer output norms for diagnostics.
    layer_norms_history: Vec<Vec<f64>>,
}

impl TransformerEncoder {
    /// Create a new transformer encoder stack.
    pub fn new(config: EncoderConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let seed = (i as u64 + 1) * 7919;
            layers.push(EncoderLayer::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.pre_norm,
                seed,
            ));
        }
        let final_norm = if config.pre_norm {
            Some(LayerNorm::new(config.d_model))
        } else {
            None
        };
        Self { config, layers, final_norm, layer_norms_history: Vec::new() }
    }

    /// Forward pass: (seq_len, d_model) -> (seq_len, d_model).
    pub fn forward(&mut self, input: &Mat) -> Mat {
        assert_eq!(input.cols, self.config.d_model, "input dim must match d_model");
        let mut hidden = input.clone();
        let mut norms = Vec::with_capacity(self.config.num_layers);

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden);
            norms.push(hidden.frobenius_norm());
        }

        if let Some(ref norm) = self.final_norm {
            hidden = norm.forward(&hidden);
        }

        self.layer_norms_history.push(norms);
        hidden
    }

    /// Get attention weights from a specific layer (after forward pass).
    pub fn layer_attention_weights(&self, layer_idx: usize) -> Option<&[Mat]> {
        self.layers.get(layer_idx).map(|l| l.attention_weights())
    }

    /// Per-layer output norms from the last forward pass.
    pub fn last_layer_norms(&self) -> Option<&[f64]> {
        self.layer_norms_history.last().map(|v| v.as_slice())
    }

    /// Total number of parameters across all layers.
    pub fn num_parameters(&self) -> usize {
        let d = self.config.d_model;
        let h = self.config.num_heads;
        let d_ff = self.config.d_ff;
        let n = self.config.num_layers;
        // Per layer: 4 attention linears + 2 FFN linears + 2 LayerNorms
        let attn_params = 4 * (d * d + d);
        let ffn_params = (d * d_ff + d_ff) + (d_ff * d + d);
        let norm_params = 2 * (2 * d); // gamma + beta for each norm
        let per_layer = attn_params + ffn_params + norm_params;
        let final_norm_params = if self.final_norm.is_some() { 2 * d } else { 0 };
        n * per_layer + final_norm_params
    }

    /// Number of layers in the stack.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Check if output norms are growing across layers (potential gradient issue).
    pub fn detect_norm_growth(&self) -> Option<bool> {
        let norms = self.last_layer_norms()?;
        if norms.len() < 2 {
            return Some(false);
        }
        let growing = norms.windows(2).all(|w| w[1] > w[0] * 1.1);
        Some(growing)
    }

    /// Reset diagnostic histories.
    pub fn reset_diagnostics(&mut self) {
        self.layer_norms_history.clear();
    }
}

impl fmt::Display for TransformerEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformerEncoder(layers={}, d_model={}, heads={}, d_ff={}, pre_norm={})",
            self.config.num_layers,
            self.config.d_model,
            self.config.num_heads,
            self.config.d_ff,
            self.config.pre_norm,
        )
    }
}

// ── Layer Summary ────────────────────────────────────────────────

/// Summary information about an encoder layer after a forward pass.
#[derive(Debug, Clone)]
pub struct LayerSummary {
    pub layer_index: usize,
    pub output_norm: f64,
    pub num_heads: usize,
    pub mean_attn_entropy: f64,
}

impl LayerSummary {
    /// Build summaries from an encoder after a forward pass.
    pub fn from_encoder(encoder: &TransformerEncoder) -> Vec<Self> {
        let norms = encoder.last_layer_norms().unwrap_or(&[]);
        let mut summaries = Vec::new();
        for (i, &norm) in norms.iter().enumerate() {
            let weights = encoder.layer_attention_weights(i);
            let mean_entropy = weights.map(|ws| {
                let mut total = 0.0;
                let mut count = 0;
                for w in ws {
                    for r in 0..w.rows {
                        let mut h = 0.0;
                        for c in 0..w.cols {
                            let p = w.get(r, c);
                            if p > 1e-12 {
                                h -= p * p.ln();
                            }
                        }
                        total += h;
                        count += 1;
                    }
                }
                if count > 0 { total / count as f64 } else { 0.0 }
            }).unwrap_or(0.0);

            summaries.push(LayerSummary {
                layer_index: i,
                output_norm: norm,
                num_heads: encoder.config.num_heads,
                mean_attn_entropy: mean_entropy,
            });
        }
        summaries
    }
}

impl fmt::Display for LayerSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Layer[{}]: norm={:.4}, heads={}, entropy={:.4}",
            self.layer_index, self.output_norm, self.num_heads, self.mean_attn_entropy
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
    fn test_mat_zeros() {
        let m = Mat::zeros(3, 4);
        assert_eq!(m.data.len(), 12);
        assert!(m.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_mat_add() {
        let a = Mat::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Mat::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_mat_frobenius_norm() {
        let m = Mat::from_data(1, 3, vec![3.0, 4.0, 0.0]);
        assert!(approx_eq(m.frobenius_norm(), 5.0, 1e-10));
    }

    #[test]
    fn test_mat_display() {
        let m = Mat::zeros(4, 8);
        assert_eq!(format!("{}", m), "Mat(4x8)");
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let ln = LayerNorm::new(4);
        let input = Mat::from_data(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let out = ln.forward(&input);
        let mean: f64 = (0..4).map(|c| out.get(0, c)).sum::<f64>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-10));
    }

    #[test]
    fn test_encoder_config_defaults() {
        let cfg = EncoderConfig::new(64, 8, 6);
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_layers, 6);
        assert_eq!(cfg.d_ff, 256);
        assert!(cfg.pre_norm);
    }

    #[test]
    fn test_encoder_config_builder() {
        let cfg = EncoderConfig::new(32, 4, 3)
            .with_d_ff(128)
            .with_dropout(0.1)
            .with_pre_norm(false);
        assert_eq!(cfg.d_ff, 128);
        assert!(!cfg.pre_norm);
    }

    #[test]
    fn test_encoder_output_shape() {
        let cfg = EncoderConfig::new(8, 2, 2);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(4, 8, vec![0.1; 32]);
        let out = enc.forward(&input);
        assert_eq!(out.rows, 4);
        assert_eq!(out.cols, 8);
    }

    #[test]
    fn test_encoder_single_layer() {
        let cfg = EncoderConfig::new(8, 2, 1);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(3, 8, vec![0.5; 24]);
        let out = enc.forward(&input);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 8);
    }

    #[test]
    fn test_encoder_post_norm() {
        let cfg = EncoderConfig::new(8, 2, 2).with_pre_norm(false);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(3, 8, vec![0.2; 24]);
        let out = enc.forward(&input);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 8);
    }

    #[test]
    fn test_encoder_attention_weights() {
        let cfg = EncoderConfig::new(8, 2, 2);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(4, 8, vec![0.1; 32]);
        enc.forward(&input);
        let weights = enc.layer_attention_weights(0).unwrap();
        assert_eq!(weights.len(), 2); // 2 heads
        assert_eq!(weights[0].rows, 4); // seq_len
    }

    #[test]
    fn test_encoder_layer_norms() {
        let cfg = EncoderConfig::new(8, 2, 3);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(4, 8, vec![0.1; 32]);
        enc.forward(&input);
        let norms = enc.last_layer_norms().unwrap();
        assert_eq!(norms.len(), 3);
        for &n in norms {
            assert!(n >= 0.0);
        }
    }

    #[test]
    fn test_encoder_num_layers() {
        let cfg = EncoderConfig::new(16, 4, 6);
        let enc = TransformerEncoder::new(cfg);
        assert_eq!(enc.num_layers(), 6);
    }

    #[test]
    fn test_encoder_num_parameters() {
        let cfg = EncoderConfig::new(8, 2, 1);
        let enc = TransformerEncoder::new(cfg);
        let params = enc.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_encoder_detect_norm_growth() {
        let cfg = EncoderConfig::new(8, 2, 2);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(3, 8, vec![0.1; 24]);
        enc.forward(&input);
        let growing = enc.detect_norm_growth();
        assert!(growing.is_some());
    }

    #[test]
    fn test_encoder_reset_diagnostics() {
        let cfg = EncoderConfig::new(8, 2, 2);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(3, 8, vec![0.1; 24]);
        enc.forward(&input);
        enc.reset_diagnostics();
        assert!(enc.last_layer_norms().is_none());
    }

    #[test]
    fn test_encoder_display() {
        let cfg = EncoderConfig::new(64, 8, 6);
        let enc = TransformerEncoder::new(cfg);
        let s = format!("{}", enc);
        assert!(s.contains("layers=6"));
        assert!(s.contains("d_model=64"));
        assert!(s.contains("heads=8"));
    }

    #[test]
    fn test_layer_summary() {
        let cfg = EncoderConfig::new(8, 2, 2);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(3, 8, vec![0.1; 24]);
        enc.forward(&input);
        let summaries = LayerSummary::from_encoder(&enc);
        assert_eq!(summaries.len(), 2);
        for s in &summaries {
            assert_eq!(s.num_heads, 2);
        }
    }

    #[test]
    fn test_layer_summary_display() {
        let s = LayerSummary {
            layer_index: 0,
            output_norm: 3.5,
            num_heads: 4,
            mean_attn_entropy: 1.2,
        };
        let disp = format!("{}", s);
        assert!(disp.contains("Layer[0]"));
        assert!(disp.contains("3.5"));
    }

    #[test]
    fn test_multiple_forward_passes() {
        let cfg = EncoderConfig::new(8, 2, 2);
        let mut enc = TransformerEncoder::new(cfg);
        let input = Mat::from_data(3, 8, vec![0.1; 24]);
        enc.forward(&input);
        enc.forward(&input);
        assert_eq!(enc.layer_norms_history.len(), 2);
    }
}
