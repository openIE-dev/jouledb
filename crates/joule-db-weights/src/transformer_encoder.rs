//! Full Multi-Layer Transformer Encoder + Pure-Rust `WordPieceTokenizer` Inside `JouleSynapse`
//!
//! Executes complete BERT / `MiniLM` / `BGE` / `E5` multi-layer bidirectional Transformer encoders
//! (`WordPiece` tokenization -> sub-1-bit `LittleBit-2` embedding row lookup -> $L$ layers of
//! Multi-Head Scaled Dot-Product Self-Attention + `GELU` FFN + `LayerNorm` -> Mean-Pooling + unit
//! L2 normalization) **100% directly from SQL-queryable sub-1-bit `LittleBitSVFTensor` bitplanes
//! and `vectors_1d`** — with **zero** PyTorch, ONNX Runtime (`ort`), `candle`, or frozen
//! `.safetensors` files at runtime!

use crate::synapse::{JouleHarnessEngine, LittleBitSVFTensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Pure-Rust BERT / MiniLM `WordPieceTokenizer` supporting `vocab.txt`, `tokenizer.json`, and
/// deterministic hash-subword fallback when no vocabulary file is present.
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    token_to_id: HashMap<String, usize>,
    vocab_size: usize,
    cls_id: usize,
    sep_id: usize,
    unk_id: usize,
    has_explicit_vocab: bool,
}

impl WordPieceTokenizer {
    /// Construct from an ordered list of vocabulary tokens (lines of HuggingFace `vocab.txt`).
    pub fn from_vocab_list(tokens: &[String]) -> Self {
        let mut map = HashMap::with_capacity(tokens.len());
        for (idx, tok) in tokens.iter().enumerate() {
            map.insert(tok.clone(), idx);
        }
        let cls_id = map.get("[CLS]").copied().unwrap_or(101);
        let sep_id = map.get("[SEP]").copied().unwrap_or(102);
        let unk_id = map.get("[UNK]").copied().unwrap_or(100);
        let vsize = tokens.len().max(256);
        Self {
            token_to_id: map,
            vocab_size: vsize,
            cls_id: cls_id.min(vsize - 1),
            sep_id: sep_id.min(vsize - 1),
            unk_id: unk_id.min(vsize - 1),
            has_explicit_vocab: !tokens.is_empty(),
        }
    }

    /// Parse standard HuggingFace `vocab.txt` content (`1` token per line).
    pub fn from_vocab_txt(raw: &str) -> Self {
        let list: Vec<String> = raw.lines().map(|l| l.trim_end_matches('\r').to_string()).collect();
        Self::from_vocab_list(&list)
    }

    /// Parse HuggingFace `tokenizer.json` (`{"model": {"vocab": {"[PAD]": 0, ...}}}`).
    pub fn from_tokenizer_json(raw_json: &str) -> Result<Self, String> {
        let val: serde_json::Value =
            serde_json::from_str(raw_json).map_err(|e| format!("invalid tokenizer.json: {e}"))?;
        let vocab_obj = val
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| "tokenizer.json missing model.vocab object".to_string())?;
        let mut max_id = 0usize;
        let mut map = HashMap::with_capacity(vocab_obj.len());
        for (tok, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                let idx = id as usize;
                max_id = max_id.max(idx);
                map.insert(tok.clone(), idx);
            }
        }
        let vsize = (max_id + 1).max(256);
        let cls_id = map.get("[CLS]").copied().unwrap_or(101).min(vsize - 1);
        let sep_id = map.get("[SEP]").copied().unwrap_or(102).min(vsize - 1);
        let unk_id = map.get("[UNK]").copied().unwrap_or(100).min(vsize - 1);
        Ok(Self {
            token_to_id: map,
            vocab_size: vsize,
            cls_id,
            sep_id,
            unk_id,
            has_explicit_vocab: true,
        })
    }

    /// Deterministic fallback tokenizer mapped onto `0..vocab_size` when no `vocab.txt` was loaded.
    pub fn fallback_for_vocab_size(vocab_size: usize) -> Self {
        let vsize = vocab_size.max(8);
        Self {
            token_to_id: HashMap::new(),
            vocab_size: vsize,
            cls_id: 101.min(vsize - 1),
            sep_id: 102.min(vsize - 1),
            unk_id: 100.min(vsize - 1),
            has_explicit_vocab: false,
        }
    }

    /// Encode text into `[CLS, tok_1, ..., tok_k, SEP]` capped at `max_seq_len`.
    pub fn encode(&self, text: &str, max_seq_len: usize) -> Vec<usize> {
        let cap = max_seq_len.clamp(4, 512);
        let mut ids = Vec::with_capacity(32);
        ids.push(self.cls_id);

        let basic_words = basic_bert_tokenize_lower(text);
        for word in basic_words {
            if ids.len() + 1 >= cap {
                break;
            }
            if !self.has_explicit_vocab {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in word.as_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
                let span = self.vocab_size.saturating_sub(106).max(1);
                let id = (106 + (h as usize % span)).min(self.vocab_size - 1);
                ids.push(id);
                continue;
            }

            // Greedy longest-match-first WordPiece decomposition
            let chars: Vec<char> = word.chars().collect();
            if chars.len() > 100 {
                ids.push(self.unk_id);
                continue;
            }
            let mut start = 0usize;
            let mut sub_tokens = Vec::new();
            let mut is_bad = false;

            while start < chars.len() {
                let mut end = chars.len();
                let mut cur_id: Option<usize> = None;
                while start < end {
                    let slice: String = chars[start..end].iter().collect();
                    let cand = if start > 0 {
                        format!("##{slice}")
                    } else {
                        slice
                    };
                    if let Some(&tid) = self.token_to_id.get(&cand) {
                        cur_id = Some(tid);
                        break;
                    }
                    end -= 1;
                }
                if let Some(tid) = cur_id {
                    sub_tokens.push(tid);
                    start = end;
                } else {
                    is_bad = true;
                    break;
                }
            }

            if is_bad {
                ids.push(self.unk_id);
            } else {
                for tid in sub_tokens {
                    if ids.len() + 1 >= cap {
                        break;
                    }
                    ids.push(tid);
                }
            }
        }

        ids.push(self.sep_id);
        ids
    }
}

fn basic_bert_tokenize_lower(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_control() {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
        } else if ch.is_ascii_punctuation() {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
            tokens.push(ch.to_string());
        } else {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

/// Receipt from executing a full multi-layer Transformer encoder pass over sub-1-bit `LittleBit-2` weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerEncoderReceipt {
    pub model_id: String,
    pub num_layers: usize,
    pub num_heads: usize,
    pub hidden_dim: usize,
    pub token_ids: Vec<usize>,
    pub embedding: Vec<f32>,
    pub avg_bits_per_weight: f64,
    pub elapsed_ns: u64,
    pub linear_projections_executed: usize,
}

/// Reconstruct a single embedding row `row_idx` directly from a sub-1-bit [`LittleBitSVFTensor`]
/// without materializing the full `[vocab_size x hidden_dim]` matrix!
pub fn reconstruct_embedding_row(tensor: &LittleBitSVFTensor, row_idx: usize) -> Vec<f32> {
    let r = row_idx.min(tensor.rows.saturating_sub(1));
    let eff_sigma = tensor.effective_latent_scales(&[]);
    let u_row = &tensor.u_bits[r * tensor.rank_words..(r + 1) * tensor.rank_words];
    let r_scale = tensor.row_scale.get(r).copied().unwrap_or(1.0);

    let mut out = vec![0.0f32; tensor.cols];
    for (j, out_j) in out.iter_mut().enumerate() {
        let v_row = &tensor.v_bits[j * tensor.rank_words..(j + 1) * tensor.rank_words];
        let mut acc = 0.0f32;
        for (k, &sig) in eff_sigma.iter().enumerate() {
            let ub = (u_row[k / 64] >> (k % 64)) & 1;
            let vb = (v_row[k / 64] >> (k % 64)) & 1;
            if ub == vb {
                acc += sig;
            } else {
                acc -= sig;
            }
        }
        *out_j = acc * r_scale * tensor.col_scale.get(j).copied().unwrap_or(1.0);
    }
    for o in &tensor.outliers {
        if o.row as usize == r && (o.col as usize) < tensor.cols {
            out[o.col as usize] += o.delta;
        }
    }
    out
}

fn layer_norm_in_place(x: &mut [f32], gamma: Option<&[f32]>, beta: Option<&[f32]>, eps: f32) {
    if x.is_empty() {
        return;
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    for (i, val) in x.iter_mut().enumerate() {
        let norm = (*val - mean) * inv_std;
        let g = gamma.and_then(|gv| gv.get(i)).copied().unwrap_or(1.0);
        let b = beta.and_then(|bv| bv.get(i)).copied().unwrap_or(0.0);
        *val = norm * g + b;
    }
}

fn gelu(x: f32) -> f32 {
    // Exact Hendrycks & Gimpel tanh GELU approximation used by BERT / MiniLM
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

fn infer_num_heads(hidden_dim: usize) -> usize {
    if hidden_dim == 384 {
        12 // MiniLM-L6 / L12 standard (384 / 12 = 32 head_dim)
    } else if hidden_dim >= 64 && hidden_dim.is_multiple_of(64) {
        hidden_dim / 64 // BERT-base (768/64=12), BERT-large (1024/64=16)
    } else if hidden_dim >= 64 && hidden_dim.is_multiple_of(32) {
        hidden_dim / 32
    } else if hidden_dim >= 16 && hidden_dim.is_multiple_of(4) {
        4
    } else {
        1
    }
}

/// Discover whether `harness` holds a complete BERT/MiniLM-family Transformer encoder under `model_id`
/// (with or without `"bert."` prefix) and return `(prefix, num_layers)`.
pub fn detect_transformer_encoder(
    harness: &JouleHarnessEngine,
    model_id: &str,
) -> Option<(String, usize)> {
    for pfx in [format!("{model_id}::"), format!("{model_id}::bert.")] {
        let word_key = format!("{pfx}embeddings.word_embeddings.weight");
        if !harness.littlebit_tensors.contains_key(&word_key) {
            continue;
        }
        let mut layers = 0usize;
        while harness
            .littlebit_tensors
            .contains_key(&format!("{pfx}encoder.layer.{layers}.attention.self.query.weight"))
        {
            layers += 1;
        }
        if layers > 0 {
            return Some((pfx, layers));
        }
    }
    None
}

/// Execute a full $L$-layer bidirectional Transformer encoder pass (`WordPiece` -> embeddings ->
/// Multi-Head Self-Attention + `GELU` FFN + `LayerNorm` -> Mean-Pooling + L2 normalization)
/// directly over sub-1-bit `LittleBitSVFTensor` bitplanes!
///
/// Callers may optionally supply `custom_linear_fn` (e.g. hardware `wgpu` Metal/Vulkan `POPCNT`
/// shader dispatch) to accelerate 2D `LittleBitSVFTensor` projections; when `None` or returning
/// `None`, falls back cleanly to exact Dual-SVID binary latent `forward_vec`.
pub fn encode_text_with_littlebit_transformer<F>(
    harness: &JouleHarnessEngine,
    model_id: &str,
    text: &str,
    max_seq_len: usize,
    mut custom_linear_fn: F,
) -> Result<TransformerEncoderReceipt, String>
where
    F: FnMut(&str, &LittleBitSVFTensor, &[f32]) -> Option<Vec<f32>>,
{
    let t0 = Instant::now();
    let (pfx, num_layers) = detect_transformer_encoder(harness, model_id).ok_or_else(|| {
        format!("model '{model_id}' does not contain BERT/MiniLM encoder layers")
    })?;

    let word_emb = harness
        .littlebit_tensors
        .get(&format!("{pfx}embeddings.word_embeddings.weight"))
        .ok_or_else(|| "missing word_embeddings.weight".to_string())?;
    let hidden_dim = word_emb.cols;
    let num_heads = infer_num_heads(hidden_dim);
    let head_dim = (hidden_dim / num_heads).max(1);

    let tokenizer = match harness.model_vocabs.get(model_id) {
        Some(vocab_vec) if !vocab_vec.is_empty() => WordPieceTokenizer::from_vocab_list(vocab_vec),
        _ => WordPieceTokenizer::fallback_for_vocab_size(word_emb.rows),
    };
    let token_ids = tokenizer.encode(text, max_seq_len);
    let seq_len = token_ids.len();

    let pos_emb = harness
        .littlebit_tensors
        .get(&format!("{pfx}embeddings.position_embeddings.weight"));
    let type_emb_2d = harness
        .littlebit_tensors
        .get(&format!("{pfx}embeddings.token_type_embeddings.weight"));
    let type_zero_row = type_emb_2d.map(|te| reconstruct_embedding_row(te, 0));

    let emb_ln_w = harness
        .vectors_1d
        .get(&format!("{pfx}embeddings.LayerNorm.weight"))
        .map(Vec::as_slice);
    let emb_ln_b = harness
        .vectors_1d
        .get(&format!("{pfx}embeddings.LayerNorm.bias"))
        .map(Vec::as_slice);

    // 1. Construct initial token + position + segment embeddings -> LayerNorm
    let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    for (pos, &tid) in token_ids.iter().enumerate() {
        let mut h = reconstruct_embedding_row(word_emb, tid);
        if let Some(pe) = pos_emb {
            let p_vec = reconstruct_embedding_row(pe, pos);
            for (dst, &pv) in h.iter_mut().zip(p_vec.iter()) {
                *dst += pv;
            }
        }
        if let Some(ref tv) = type_zero_row {
            for (dst, &tval) in h.iter_mut().zip(tv.iter()) {
                *dst += tval;
            }
        }
        layer_norm_in_place(&mut h, emb_ln_w, emb_ln_b, 1e-12);
        hidden_states.push(h);
    }

    let mut bpw_sum = word_emb.bits_per_weight();
    let mut bpw_cnt = 1usize;
    let mut linear_ops = 0usize;

    let mut run_linear = |fq_key: &str, x: &[f32], bias_key: &str| -> Result<Vec<f32>, String> {
        let tensor = harness
            .littlebit_tensors
            .get(fq_key)
            .ok_or_else(|| format!("missing encoder tensor '{fq_key}'"))?;
        bpw_sum += tensor.bits_per_weight();
        bpw_cnt += 1;
        linear_ops += 1;

        let mut y = match custom_linear_fn(fq_key, tensor, x) {
            Some(gpu_y) => gpu_y,
            None => tensor.forward_vec(x, &[])?,
        };
        if let Some(b) = harness.vectors_1d.get(bias_key) {
            for (yi, &bi) in y.iter_mut().zip(b.iter()) {
                *yi += bi;
            }
        }
        Ok(y)
    };

    // 2. Execute all L Transformer Encoder Layers
    let inv_sqrt_dh = 1.0 / (head_dim as f32).sqrt();
    for l in 0..num_layers {
        let lp = format!("{pfx}encoder.layer.{l}");
        let mut q_seq = Vec::with_capacity(seq_len);
        let mut k_seq = Vec::with_capacity(seq_len);
        let mut v_seq = Vec::with_capacity(seq_len);

        for h in &hidden_states {
            q_seq.push(run_linear(
                &format!("{lp}.attention.self.query.weight"),
                h,
                &format!("{lp}.attention.self.query.bias"),
            )?);
            k_seq.push(run_linear(
                &format!("{lp}.attention.self.key.weight"),
                h,
                &format!("{lp}.attention.self.key.bias"),
            )?);
            v_seq.push(run_linear(
                &format!("{lp}.attention.self.value.weight"),
                h,
                &format!("{lp}.attention.self.value.bias"),
            )?);
        }

        // Multi-Head Scaled Dot-Product Self-Attention
        let mut attn_concat_seq = vec![vec![0.0f32; hidden_dim]; seq_len];
        for head in 0..num_heads {
            let h_off = head * head_dim;
            let h_end = (h_off + head_dim).min(hidden_dim);
            for t in 0..seq_len {
                let qt = &q_seq[t][h_off..h_end];
                let mut scores = Vec::with_capacity(seq_len);
                let mut max_s = f32::NEG_INFINITY;
                for k_item in k_seq.iter().take(seq_len) {
                    let ks = &k_item[h_off..h_end];
                    let dot: f32 = qt.iter().zip(ks.iter()).map(|(a, b)| a * b).sum();
                    let sc = dot * inv_sqrt_dh;
                    if sc > max_s {
                        max_s = sc;
                    }
                    scores.push(sc);
                }
                let mut denom = 0.0f32;
                for sc in &mut scores {
                    *sc = (*sc - max_s).exp();
                    denom += *sc;
                }
                let inv_denom = 1.0 / denom.max(1e-12);
                let dst_slice = &mut attn_concat_seq[t][h_off..h_end];
                for (s, &prob_unnorm) in scores.iter().enumerate() {
                    let p = prob_unnorm * inv_denom;
                    let vs = &v_seq[s][h_off..h_end];
                    for (d, &vval) in dst_slice.iter_mut().zip(vs.iter()) {
                        *d += p * vval;
                    }
                }
            }
        }

        let attn_ln_w = harness
            .vectors_1d
            .get(&format!("{lp}.attention.output.LayerNorm.weight"))
            .map(Vec::as_slice);
        let attn_ln_b = harness
            .vectors_1d
            .get(&format!("{lp}.attention.output.LayerNorm.bias"))
            .map(Vec::as_slice);
        let out_ln_w = harness
            .vectors_1d
            .get(&format!("{lp}.output.LayerNorm.weight"))
            .map(Vec::as_slice);
        let out_ln_b = harness
            .vectors_1d
            .get(&format!("{lp}.output.LayerNorm.bias"))
            .map(Vec::as_slice);

        // Attention output projection + residual + LayerNorm -> FFN (GELU) -> residual + LayerNorm
        for t in 0..seq_len {
            let attn_proj = run_linear(
                &format!("{lp}.attention.output.dense.weight"),
                &attn_concat_seq[t],
                &format!("{lp}.attention.output.dense.bias"),
            )?;
            for (hi, &ai) in hidden_states[t].iter_mut().zip(attn_proj.iter()) {
                *hi += ai;
            }
            layer_norm_in_place(&mut hidden_states[t], attn_ln_w, attn_ln_b, 1e-12);

            let mut ffn_mid = run_linear(
                &format!("{lp}.intermediate.dense.weight"),
                &hidden_states[t],
                &format!("{lp}.intermediate.dense.bias"),
            )?;
            for v in &mut ffn_mid {
                *v = gelu(*v);
            }
            let ffn_out = run_linear(
                &format!("{lp}.output.dense.weight"),
                &ffn_mid,
                &format!("{lp}.output.dense.bias"),
            )?;
            for (hi, &fi) in hidden_states[t].iter_mut().zip(ffn_out.iter()) {
                *hi += fi;
            }
            layer_norm_in_place(&mut hidden_states[t], out_ln_w, out_ln_b, 1e-12);
        }
    }

    // 3. Mean-Pooling across sequence tokens + unit L2 normalization
    let mut pooled = vec![0.0f32; hidden_dim];
    let inv_seq = 1.0 / (seq_len.max(1) as f32);
    for h in &hidden_states {
        for (pi, &hi) in pooled.iter_mut().zip(h.iter()) {
            *pi += hi * inv_seq;
        }
    }
    let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    for pi in &mut pooled {
        *pi /= norm;
    }

    Ok(TransformerEncoderReceipt {
        model_id: model_id.to_string(),
        num_layers,
        num_heads,
        hidden_dim,
        token_ids,
        embedding: pooled,
        avg_bits_per_weight: bpw_sum / (bpw_cnt.max(1) as f64),
        elapsed_ns: (t0.elapsed().as_nanos() as u64).max(1),
        linear_projections_executed: linear_ops,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_ingest::{
        SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes,
        ingest_safetensors_into_harness,
    };
    use std::collections::BTreeMap;

    #[test]
    fn test_wordpiece_tokenizer_and_full_multilayer_littlebit_transformer_encoder() {
        let vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\nthermodynamic\ndatabase\njoule\n##s\nenergy\n";
        let tok = WordPieceTokenizer::from_vocab_txt(vocab);
        let ids = tok.encode("Thermodynamic Joules!", 32);
        // [CLS]=2, thermodynamic=4, joule=6, ##s=7, !=[UNK]=1, [SEP]=3
        assert_eq!(ids, vec![2, 4, 6, 7, 1, 3]);

        // Build a synthetic 2-layer, 16-dim (4-head), 32-FFN BERT/MiniLM safetensors model
        let d = 16usize;
        let ffn = 32usize;
        let mk_2d = |name: &str, r: usize, c: usize, seed: f32| SafetensorBuildEntry {
            name: name.to_string(),
            dtype: SafetensorDtype::F32,
            shape: vec![r, c],
            values_f32: (0..(r * c))
                .map(|i| ((i as f32) * 0.13 + seed).sin() * 0.25)
                .collect(),
        };
        let mk_1d = |name: &str, n: usize, val: f32| SafetensorBuildEntry {
            name: name.to_string(),
            dtype: SafetensorDtype::F32,
            shape: vec![n],
            values_f32: vec![val; n],
        };

        let mut entries = vec![
            mk_2d("embeddings.word_embeddings.weight", 32, d, 0.1),
            mk_2d("embeddings.position_embeddings.weight", 32, d, 0.2),
            mk_2d("embeddings.token_type_embeddings.weight", 2, d, 0.0),
            mk_1d("embeddings.LayerNorm.weight", d, 1.0),
            mk_1d("embeddings.LayerNorm.bias", d, 0.0),
        ];
        for l in 0..2 {
            let p = format!("encoder.layer.{l}");
            entries.extend([
                mk_2d(&format!("{p}.attention.self.query.weight"), d, d, 0.3 + l as f32),
                mk_1d(&format!("{p}.attention.self.query.bias"), d, 0.0),
                mk_2d(&format!("{p}.attention.self.key.weight"), d, d, 0.4 + l as f32),
                mk_1d(&format!("{p}.attention.self.key.bias"), d, 0.0),
                mk_2d(&format!("{p}.attention.self.value.weight"), d, d, 0.5 + l as f32),
                mk_1d(&format!("{p}.attention.self.value.bias"), d, 0.0),
                mk_2d(&format!("{p}.attention.output.dense.weight"), d, d, 0.6 + l as f32),
                mk_1d(&format!("{p}.attention.output.dense.bias"), d, 0.0),
                mk_1d(&format!("{p}.attention.output.LayerNorm.weight"), d, 1.0),
                mk_1d(&format!("{p}.attention.output.LayerNorm.bias"), d, 0.0),
                mk_2d(&format!("{p}.intermediate.dense.weight"), ffn, d, 0.7 + l as f32),
                mk_1d(&format!("{p}.intermediate.dense.bias"), ffn, 0.0),
                mk_2d(&format!("{p}.output.dense.weight"), d, ffn, 0.8 + l as f32),
                mk_1d(&format!("{p}.output.dense.bias"), d, 0.0),
                mk_1d(&format!("{p}.output.LayerNorm.weight"), d, 1.0),
                mk_1d(&format!("{p}.output.LayerNorm.bias"), d, 0.0),
            ]);
        }

        let raw = build_safetensors_bytes(&BTreeMap::new(), &entries).unwrap();
        let mut harness = JouleHarnessEngine::new(16, 16, 16 * 1024 * 1024).unwrap();
        ingest_safetensors_into_harness("mini-enc", &raw, 8, 4, &mut harness).unwrap();
        harness.model_vocabs.insert(
            "mini-enc".into(),
            vocab.lines().map(String::from).collect(),
        );

        let rec_a = encode_text_with_littlebit_transformer(
            &harness,
            "mini-enc",
            "thermodynamic joules",
            32,
            |_, _, _| None,
        )
        .unwrap();
        let rec_b = encode_text_with_littlebit_transformer(
            &harness,
            "mini-enc",
            "thermodynamic database energy",
            32,
            |_, _, _| None,
        )
        .unwrap();

        assert_eq!(rec_a.num_layers, 2);
        assert_eq!(rec_a.hidden_dim, 16);
        assert_eq!(rec_a.embedding.len(), 16);
        // Unit L2 normalized
        let l2: f32 = rec_a.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((l2 - 1.0).abs() < 1e-4);
        // Distinct texts yield distinct non-degenerate embeddings
        let cos: f32 = rec_a
            .embedding
            .iter()
            .zip(rec_b.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(cos.is_finite() && cos < 0.9999);
    }
}
