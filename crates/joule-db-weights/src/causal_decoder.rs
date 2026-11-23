//! Full Causal Autoregressive Transformer Decoder (`RoPE` + `GQA` + `RMSNorm` + `SwiGLU` +
//! Pure-Rust Byte-Level `BpeTokenizer` + Materialized `Mooncake` Prefix KV-Cache) Inside `JouleSynapse`.
//!
//! Executes `Llama-3` / `SmolLM2` / `Qwen2.5` / `Mistral`-family causal language models directly
//! from SQL-materialized sub-1-bit [`LittleBitSVFTensor`] bitplanes and `vectors_1d` — with **zero**
//! `llama.cpp`, `vLLM`, `Ollama`, PyTorch, ONNX Runtime (`ort`), or `candle` dependencies!

use crate::synapse::{
    CausalModelMeta, JouleHarnessEngine, KvPrefixMatchReceipt, KvStorageTier, LittleBitSVFTensor,
};
use crate::transformer_encoder::reconstruct_embedding_row;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

const MAX_LIVE_KV_BLOCKS: usize = 256;

/// Build the reversible GPT-2 / Llama-3 / Qwen2 / SmolLM2 `byte -> unicode char` table (`256` entries).
pub fn gpt2_byte_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    for b in b'!'..=b'~' {
        bs.push(b);
    }
    for b in 0xA1u8..=0xACu8 {
        bs.push(b);
    }
    for b in 0xAEu8..=0xFFu8 {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u16..=255u16 {
        let byte = b as u8;
        if !bs.contains(&byte) {
            bs.push(byte);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut fwd = ['\0'; 256];
    let mut rev = HashMap::with_capacity(256);
    for (&b, &code) in bs.iter().zip(cs.iter()) {
        if let Some(ch) = char::from_u32(code) {
            fwd[b as usize] = ch;
            rev.insert(ch, b);
        }
    }
    (fwd, rev)
}

/// Pure-Rust Byte-Level & SentencePiece Metaspace `BpeTokenizer` (`GPT-2`, `Llama-3`, `SmolLM2`,
/// `Qwen2.5`/`Qwen3`, `LiquidAI LFM2`/`LFM3`, `Gemma 4`, `Mistral`, `PrismML Bonsai`).
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    merge_ranks: HashMap<(String, String), usize>,
    special_tokens: Vec<(String, usize)>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    vocab_size: usize,
    has_explicit_vocab: bool,
    uses_metaspace: bool,
}

impl BpeTokenizer {
    /// Construct from ordered `vocab_tokens` (`id -> token`) and ordered BPE `merges`.
    pub fn from_vocab_and_merges(vocab_tokens: &[String], merges: &[(String, String)]) -> Self {
        let (byte_to_char, char_to_byte) = gpt2_byte_to_unicode();
        let vsize = vocab_tokens.len().max(256);
        let mut token_to_id = HashMap::with_capacity(vocab_tokens.len());
        let mut special_tokens = Vec::new();
        for (idx, tok) in vocab_tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), idx);
            if (tok.starts_with("<|") && tok.ends_with("|>") && tok.len() >= 5)
                || (tok.starts_with('<')
                    && tok.ends_with('>')
                    && !tok.starts_with("<0x")
                    && tok.len() >= 3
                    && tok.len() <= 32)
            {
                special_tokens.push((tok.clone(), idx));
            }
        }
        // Match longest special tokens first
        special_tokens.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, pair) in merges.iter().enumerate() {
            merge_ranks.entry(pair.clone()).or_insert(rank);
        }
        let uses_metaspace = !token_to_id.contains_key("Ġ")
            && (token_to_id.contains_key("▁")
                || vocab_tokens.iter().take(4096).any(|t| t.starts_with('▁')));

        Self {
            token_to_id,
            id_to_token: vocab_tokens.to_vec(),
            merge_ranks,
            special_tokens,
            byte_to_char,
            char_to_byte,
            vocab_size: vsize,
            has_explicit_vocab: !vocab_tokens.is_empty(),
            uses_metaspace,
        }
    }

    /// Parse HuggingFace `tokenizer.json` (`model.vocab`, `model.merges`, and `added_tokens`) into
    /// `(ordered_vocab, bpe_merges)`.
    #[allow(clippy::type_complexity)]
    pub fn parse_hf_tokenizer_json(
        raw_json: &str,
    ) -> Result<(Vec<String>, Vec<(String, String)>), String> {
        let val: serde_json::Value =
            serde_json::from_str(raw_json).map_err(|e| format!("invalid tokenizer.json: {e}"))?;
        let model = val
            .get("model")
            .ok_or_else(|| "tokenizer.json missing 'model'".to_string())?;
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "tokenizer.json missing 'model.vocab'".to_string())?;

        let mut max_id = 0usize;
        for id_v in vocab_obj.values() {
            if let Some(id) = id_v.as_u64() {
                max_id = max_id.max(id as usize);
            }
        }
        if let Some(added) = val.get("added_tokens").and_then(|a| a.as_array()) {
            for entry in added {
                if let Some(id) = entry.get("id").and_then(|i| i.as_u64()) {
                    max_id = max_id.max(id as usize);
                }
            }
        }
        if max_id > 512_000 {
            return Err(format!("vocab max token id {max_id} exceeds safety bound"));
        }

        let mut vocab = vec![String::new(); max_id + 1];
        for (tok, id_v) in vocab_obj {
            if let Some(id) = id_v.as_u64() {
                vocab[id as usize] = tok.clone();
            }
        }
        if let Some(added) = val.get("added_tokens").and_then(|a| a.as_array()) {
            for entry in added {
                if let (Some(id), Some(content)) = (
                    entry.get("id").and_then(|i| i.as_u64()),
                    entry.get("content").and_then(|c| c.as_str()),
                ) {
                    vocab[id as usize] = content.to_string();
                }
            }
        }

        let mut merges = Vec::new();
        if let Some(merge_arr) = model.get("merges").and_then(|m| m.as_array()) {
            merges.reserve(merge_arr.len());
            for item in merge_arr {
                if let Some(s) = item.as_str() {
                    if let Some((l, r)) = s.split_once(' ') {
                        merges.push((l.to_string(), r.to_string()));
                    }
                } else if let Some(pair) = item.as_array()
                    && pair.len() == 2
                    && let (Some(l), Some(r)) = (pair[0].as_str(), pair[1].as_str())
                {
                    merges.push((l.to_string(), r.to_string()));
                }
            }
        }

        Ok((vocab, merges))
    }

    /// Fallback hash tokenizer when no explicit vocabulary is loaded.
    pub fn fallback_for_vocab_size(vocab_size: usize) -> Self {
        let mut s = Self::from_vocab_and_merges(&[], &[]);
        s.vocab_size = vocab_size.max(16);
        s.has_explicit_vocab = false;
        s
    }

    /// Encode UTF-8 text into BPE token IDs (capped at `max_tokens`).
    pub fn encode(&self, text: &str, max_tokens: usize) -> Vec<usize> {
        let cap = max_tokens.clamp(1, 4096);
        if !self.has_explicit_vocab {
            let mut out = Vec::new();
            for word in text.split_whitespace().take(cap) {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in word.as_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
                out.push((h as usize) % self.vocab_size);
            }
            if out.is_empty() {
                out.push(1 % self.vocab_size);
            }
            return out;
        }

        let mut ids = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() && ids.len() < cap {
            let rem = &text[cursor..];
            if let Some((spec_str, spec_id)) = self
                .special_tokens
                .iter()
                .find(|(st, _)| rem.starts_with(st.as_str()))
            {
                ids.push(*spec_id);
                cursor += spec_str.len();
                continue;
            }

            // Find next special token boundary (if any)
            let mut next_spec_off = rem.len();
            for (st, _) in &self.special_tokens {
                if let Some(pos) = rem.find(st.as_str())
                    && pos < next_spec_off
                {
                    next_spec_off = pos;
                }
            }
            let chunk = &rem[..next_spec_off];
            for pre_tok in gpt2_pretokenize(chunk) {
                if ids.len() >= cap {
                    break;
                }
                self.encode_pretoken(pre_tok, cap, &mut ids);
            }
            cursor += next_spec_off;
        }
        if ids.is_empty() {
            ids.push(0);
        }
        ids
    }

    fn encode_pretoken(&self, piece: &str, cap: usize, out: &mut Vec<usize>) {
        // Fast exact match (works for whitespace/word vocabularies without BPE merges too!)
        if let Some(&direct_id) = self.token_to_id.get(piece) {
            out.push(direct_id);
            return;
        }
        let mapped: Vec<String> = if self.uses_metaspace {
            piece
                .chars()
                .map(|c| {
                    if c == ' ' {
                        "▁".to_string()
                    } else {
                        c.to_string()
                    }
                })
                .collect()
        } else {
            piece
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize].to_string())
                .collect()
        };
        if mapped.is_empty() {
            return;
        }
        let joined: String = mapped.concat();
        if let Some(&direct_mapped_id) = self.token_to_id.get(&joined) {
            out.push(direct_mapped_id);
            return;
        }

        let mut parts = mapped;
        while parts.len() >= 2 && !self.merge_ranks.is_empty() {
            let mut best_rank = usize::MAX;
            let mut best_pair: Option<(String, String)> = None;
            for w in parts.windows(2) {
                let cand = (w[0].clone(), w[1].clone());
                if let Some(&r) = self.merge_ranks.get(&cand)
                    && r < best_rank
                {
                    best_rank = r;
                    best_pair = Some(cand);
                }
            }
            let Some((left, right)) = best_pair else {
                break;
            };
            let mut next_parts = Vec::with_capacity(parts.len());
            let mut i = 0usize;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == left && parts[i + 1] == right {
                    next_parts.push(format!("{left}{right}"));
                    i += 2;
                } else {
                    next_parts.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = next_parts;
        }

        for p in parts {
            if out.len() >= cap {
                break;
            }
            if let Some(&tid) = self.token_to_id.get(&p) {
                out.push(tid);
            } else {
                out.push(0);
            }
        }
    }

    /// Decode generated token IDs back into clean UTF-8 text (skipping special control tokens
    /// when `skip_special_tokens` is `true`).
    pub fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> String {
        if !self.has_explicit_vocab {
            return token_ids
                .iter()
                .map(|id| format!("tok_{id}"))
                .collect::<Vec<_>>()
                .join(" ");
        }
        let mut raw_bytes: Vec<u8> = Vec::with_capacity(token_ids.len() * 4);
        let mut out_str = String::new();

        for &tid in token_ids {
            let Some(tok) = self.id_to_token.get(tid) else {
                continue;
            };
            let is_spec = self.special_tokens.iter().any(|(_, sid)| *sid == tid);
            if is_spec {
                if !skip_special_tokens {
                    if !raw_bytes.is_empty() {
                        out_str.push_str(&String::from_utf8_lossy(&raw_bytes));
                        raw_bytes.clear();
                    }
                    out_str.push_str(tok);
                }
                continue;
            }
            if self.uses_metaspace {
                if !raw_bytes.is_empty() {
                    out_str.push_str(&String::from_utf8_lossy(&raw_bytes));
                    raw_bytes.clear();
                }
                out_str.push_str(&tok.replace('▁', " "));
                continue;
            }
            let mut all_mapped = true;
            let mut scratch = Vec::with_capacity(tok.len());
            for ch in tok.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    scratch.push(b);
                } else {
                    all_mapped = false;
                    break;
                }
            }
            if all_mapped {
                raw_bytes.extend_from_slice(&scratch);
            } else {
                if !raw_bytes.is_empty() {
                    out_str.push_str(&String::from_utf8_lossy(&raw_bytes));
                    raw_bytes.clear();
                }
                out_str.push_str(tok);
            }
        }
        if !raw_bytes.is_empty() {
            out_str.push_str(&String::from_utf8_lossy(&raw_bytes));
        }
        out_str
    }
}

/// GPT-2 / Llama-3 style pre-tokenizer keeping leading spaces attached to words (`" hello"`).
fn gpt2_pretokenize(s: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let start = i;
        if bytes[i] == b' '
            && i + 1 < bytes.len()
            && bytes[i + 1] != b' '
            && bytes[i + 1] != b'\n'
            && bytes[i + 1] != b'\r'
        {
            i += 1;
        }
        if i < bytes.len() && (bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b' ') {
            while i < bytes.len() && (bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b' ') {
                i += 1;
            }
        } else if i < bytes.len() && bytes[i].is_ascii_punctuation() {
            i += 1;
        } else {
            while i < bytes.len()
                && !bytes[i].is_ascii_whitespace()
                && !bytes[i].is_ascii_punctuation()
            {
                let ch_len = s[i..].chars().next().map(char::len_utf8).unwrap_or(1);
                i += ch_len;
            }
        }
        if i > start {
            spans.push(&s[start..i]);
        }
    }
    spans
}

/// Parse HuggingFace `config.json` (including nested `text_config` for VLMs / `Gemma 4` / `SmolVLM`)
/// into [`CausalModelMeta`].
pub fn parse_hf_causal_config_json(raw_json: &str) -> Result<CausalModelMeta, String> {
    let v: serde_json::Value =
        serde_json::from_str(raw_json).map_err(|e| format!("invalid config.json: {e}"))?;
    let cfg = if v.get("num_attention_heads").is_none()
        && let Some(tc) = v.get("text_config")
    {
        tc
    } else {
        &v
    };

    let mut eos_ids = Vec::new();
    for eos_src in [cfg.get("eos_token_id"), v.get("eos_token_id")] {
        if let Some(eos_v) = eos_src {
            if let Some(single) = eos_v.as_u64() {
                let id = single as usize;
                if !eos_ids.contains(&id) {
                    eos_ids.push(id);
                }
            } else if let Some(arr) = eos_v.as_array() {
                for x in arr {
                    if let Some(id) = x.as_u64() {
                        let uid = id as usize;
                        if !eos_ids.contains(&uid) {
                            eos_ids.push(uid);
                        }
                    }
                }
            }
        }
    }

    let model_type = cfg
        .get("model_type")
        .or_else(|| v.get("model_type"))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let hidden_act = cfg
        .get("hidden_act")
        .or_else(|| cfg.get("hidden_activation"))
        .and_then(|x| x.as_str())
        .unwrap_or(if model_type.contains("gemma") {
            "gelu_pytorch_tanh"
        } else if model_type.contains("bitnet") {
            "relu2"
        } else {
            "silu"
        })
        .to_ascii_lowercase();
    let rope_theta = cfg
        .get("rope_theta")
        .or_else(|| {
            cfg.get("rope_parameters")
                .and_then(|rp| rp.get("rope_theta"))
        })
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(10_000.0);
    let rms_norm_eps = cfg
        .get("rms_norm_eps")
        .or_else(|| cfg.get("norm_eps"))
        .or_else(|| cfg.get("layer_norm_epsilon"))
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(1e-5);
    let conv_l_cache = cfg
        .get("conv_L_cache")
        .or_else(|| cfg.get("conv_l_cache"))
        .and_then(|x| x.as_u64())
        .map(|x| (x as usize).max(1))
        .unwrap_or(3);

    Ok(CausalModelMeta {
        num_attention_heads: cfg
            .get("num_attention_heads")
            .or_else(|| cfg.get("decoder_attention_heads"))
            .and_then(|x| x.as_u64())
            .map(|x| x as usize),
        num_key_value_heads: cfg
            .get("num_key_value_heads")
            .or_else(|| cfg.get("num_kv_heads"))
            .or_else(|| cfg.get("num_attention_heads"))
            .or_else(|| cfg.get("decoder_attention_heads"))
            .and_then(|x| x.as_u64())
            .map(|x| x as usize),
        head_dim: cfg
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize),
        rope_theta,
        rms_norm_eps,
        bos_token_id: cfg
            .get("bos_token_id")
            .or_else(|| v.get("bos_token_id"))
            .and_then(|x| x.as_u64())
            .map(|x| x as usize),
        eos_token_ids: eos_ids,
        tie_word_embeddings: cfg
            .get("tie_word_embeddings")
            .or_else(|| v.get("tie_word_embeddings"))
            .and_then(|x| x.as_bool())
            .unwrap_or(true),
        gemma_rms_norm_offset: model_type.contains("gemma"),
        final_logit_softcapping: cfg
            .get("final_logit_softcapping")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32),
        model_type,
        hidden_act,
        conv_l_cache,
    })
}

/// Generation hyperparameters for [`generate_with_littlebit_causal_decoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGenerationParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    #[serde(default)]
    pub stop_token_ids: Vec<usize>,
    /// Number of self-speculative `LayerSkip` + rank-8 (`0.04` BPW) draft tokens proposed per
    /// batched target verification step (`0` = standard 1-token decode; `4` = 5-wide batched verify).
    #[serde(default)]
    pub speculative_draft_k: usize,
}

impl Default for CausalGenerationParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            temperature: 0.0,
            top_p: 0.9,
            repetition_penalty: 1.12,
            seed: 42,
            stop_token_ids: Vec::new(),
            speculative_draft_k: 0,
        }
    }
}

/// Receipt from executing a full multi-layer causal autoregressive Transformer generation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDecoderReceipt {
    pub model_id: String,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub generated_text: String,
    pub avg_bits_per_weight: f64,
    pub elapsed_ns: u64,
    pub tokens_per_sec: f64,
    pub linear_projections_executed: usize,
    pub kv_prefix_receipt: KvPrefixMatchReceipt,
    #[serde(default)]
    pub gpu_accelerated: bool,
    #[serde(default)]
    pub gpu_dispatches: usize,
    #[serde(default)]
    pub execution_backend: String,
    #[serde(default)]
    pub speculative_drafted_tokens: usize,
    #[serde(default)]
    pub speculative_accepted_tokens: usize,
    #[serde(default)]
    pub speculative_acceptance_rate: f64,
}

/// Materialized `(K, V, ShortConv)` block entry inside [`LivePrefixKvStore`] (`Mooncake` tensor reuse).
#[derive(Debug, Clone)]
struct CachedKvBlockTensors {
    tokens: Vec<usize>,
    /// `[num_layers][block_len][kv_dim]`
    k_layers: Vec<Vec<Vec<f32>>>,
    /// `[num_layers][block_len][kv_dim]`
    v_layers: Vec<Vec<Vec<f32>>>,
    /// `[num_layers][block_len][hidden_dim]` for hybrid `ShortConv` layers (`LiquidAI LFM2`/`LFM3`)
    conv_layers: Vec<Vec<Vec<f32>>>,
    /// Hidden output of the very last token of this block (`[hidden_dim]`), used when the entire
    /// prompt ends on an exact cached block boundary!
    last_hidden_normed: Vec<f32>,
}

/// Live materialized `(K, V)` tensor prefix pool complementing `Mooncake` [`RadixKvCachePool`] so
/// shared system/chat prompt prefixes genuinely skip prefill projections in silicon!
#[derive(Debug, Default)]
pub struct LivePrefixKvStore {
    pub block_tokens: usize,
    blocks: HashMap<(String, u64), CachedKvBlockTensors>,
    fifo: VecDeque<(String, u64)>,
}

impl LivePrefixKvStore {
    pub fn new(block_tokens: usize) -> Self {
        Self {
            block_tokens: block_tokens.clamp(4, 128),
            blocks: HashMap::new(),
            fifo: VecDeque::new(),
        }
    }

    fn chain_hash(parent: u64, chunk: &[usize]) -> u64 {
        let mut h = parent ^ 0xCBF2_9CE4_8422_2325;
        for &t in chunk {
            h ^= t as u64;
            h = h.wrapping_mul(0x0100_0000_01B3);
            h ^= h >> 31;
        }
        h
    }
}

/// Discover whether `harness` holds a causal or hybrid `ShortConv`+`GQA` decoder (`Llama`, `SmolLM2`,
/// `Qwen2.5`/`Qwen3`, `LiquidAI LFM2`/`LFM2.5`/`LFM3`, `PrismML Bonsai`, `BitNet b1.58`,
/// `Google Gemma 4`, `Cactus Needle`, `SmolVLM`, `Granite-Docling`, `Poolside Laguna`) under
/// `model_id` and return `(prefix, num_layers)`.
pub fn detect_causal_decoder(
    harness: &JouleHarnessEngine,
    model_id: &str,
) -> Option<(String, usize)> {
    for pfx in [
        format!("{model_id}::model."),
        format!("{model_id}::"),
        format!("{model_id}::model.text_model."),
        format!("{model_id}::text_model."),
        format!("{model_id}::model.language_model."),
        format!("{model_id}::language_model.model."),
        format!("{model_id}::model.decoder."),
    ] {
        let has_emb = harness
            .littlebit_tensors
            .contains_key(&format!("{pfx}embed_tokens.weight"))
            || harness
                .littlebit_tensors
                .contains_key(&format!("{model_id}::model.embed_tokens.weight"))
            || harness
                .littlebit_tensors
                .contains_key(&format!("{model_id}::embed_tokens.weight"));
        if !has_emb {
            continue;
        }
        let mut layers = 0usize;
        while harness
            .littlebit_tensors
            .contains_key(&format!("{pfx}layers.{layers}.self_attn.q_proj.weight"))
            || harness
                .littlebit_tensors
                .contains_key(&format!("{pfx}layers.{layers}.conv.in_proj.weight"))
        {
            layers += 1;
        }
        if layers > 0 {
            return Some((pfx, layers));
        }
    }
    None
}

fn resolve_embed_key(harness: &JouleHarnessEngine, model_id: &str, pfx: &str) -> Option<String> {
    [
        format!("{pfx}embed_tokens.weight"),
        format!("{model_id}::model.embed_tokens.weight"),
        format!("{model_id}::embed_tokens.weight"),
    ]
    .into_iter()
    .find(|k| harness.littlebit_tensors.contains_key(k))
}

fn find_first_attention_q_k<'a>(
    harness: &'a JouleHarnessEngine,
    pfx: &str,
    num_layers: usize,
) -> Option<(&'a LittleBitSVFTensor, &'a LittleBitSVFTensor)> {
    for l in 0..num_layers {
        if let (Some(qt), Some(kt)) = (
            harness
                .littlebit_tensors
                .get(&format!("{pfx}layers.{l}.self_attn.q_proj.weight")),
            harness
                .littlebit_tensors
                .get(&format!("{pfx}layers.{l}.self_attn.k_proj.weight")),
        ) {
            return Some((qt, kt));
        }
    }
    None
}

fn rms_norm_cfg(x: &[f32], weight: Option<&[f32]>, eps: f32, plus_one_offset: bool) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = match weight.and_then(|wv| wv.get(i)).copied() {
                Some(wi) if plus_one_offset => 1.0 + wi,
                Some(wi) => wi,
                None => 1.0,
            };
            v * inv_rms * w
        })
        .collect()
}

fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    rms_norm_cfg(x, weight, eps, false)
}

fn apply_qk_norm_in_place(
    vec: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    norm_weight: Option<&[f32]>,
    eps: f32,
    plus_one_offset: bool,
) {
    let Some(nw) = norm_weight else {
        return;
    };
    if nw.len() == head_dim && head_dim > 0 {
        for h in 0..num_heads {
            let base = h * head_dim;
            if base + head_dim <= vec.len() {
                let normed =
                    rms_norm_cfg(&vec[base..base + head_dim], Some(nw), eps, plus_one_offset);
                vec[base..base + head_dim].copy_from_slice(&normed);
            }
        }
    } else if nw.len() == vec.len() {
        let normed = rms_norm_cfg(vec, Some(nw), eps, plus_one_offset);
        vec.copy_from_slice(&normed);
    }
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

#[inline]
fn apply_gated_act(gate: f32, up: f32, act: &str) -> f32 {
    if act.contains("relu2") || act.contains("squared_relu") {
        let r = gate.max(0.0);
        r * r * up
    } else if act.contains("gelu") {
        gelu_tanh(gate) * up
    } else {
        silu(gate) * up
    }
}

fn first_vec_1d<'a>(
    vecs: &'a BTreeMap<String, Vec<f32>>,
    candidates: &[String],
) -> Option<&'a [f32]> {
    for k in candidates {
        if let Some(v) = vecs.get(k) {
            return Some(v.as_slice());
        }
    }
    None
}

use std::collections::BTreeMap;

fn apply_rope_half_split_in_place(
    vec: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    pos: usize,
    rope_theta: f32,
) {
    let half = head_dim / 2;
    if half == 0 {
        return;
    }
    for h in 0..num_heads {
        let base = h * head_dim;
        if base + head_dim > vec.len() {
            break;
        }
        let head = &mut vec[base..base + head_dim];
        for i in 0..half {
            let freq_exp = (2 * i) as f32 / (head_dim as f32);
            let inv_freq = rope_theta.powf(-freq_exp);
            let angle = (pos as f32) * inv_freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos_a - x1 * sin_a;
            head[i + half] = x1 * cos_a + x0 * sin_a;
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn infer_gqa_geometry(
    q_dim: usize,
    kv_dim: usize,
    meta: Option<&CausalModelMeta>,
) -> (usize, usize, usize) {
    if let Some(m) = meta
        && let (Some(nh), Some(nkv)) = (m.num_attention_heads, m.num_key_value_heads)
        && nh > 0
        && nkv > 0
    {
        let hd = m.head_dim.unwrap_or_else(|| (q_dim / nh).max(1));
        return (nh, nkv, hd);
    }
    let hd = if kv_dim >= 64 && kv_dim.is_multiple_of(64) && q_dim.is_multiple_of(64) {
        64
    } else if kv_dim >= 32 && kv_dim.is_multiple_of(32) && q_dim.is_multiple_of(32) {
        32
    } else if kv_dim >= 16 && kv_dim.is_multiple_of(16) && q_dim.is_multiple_of(16) {
        16
    } else if kv_dim >= 8 && kv_dim.is_multiple_of(8) && q_dim.is_multiple_of(8) {
        8
    } else if kv_dim >= 4 && kv_dim.is_multiple_of(4) && q_dim.is_multiple_of(4) {
        4
    } else {
        kv_dim.max(1)
    };
    let nh = (q_dim / hd).max(1);
    let nkv = (kv_dim / hd).max(1);
    (nh, nkv, hd)
}

fn sample_next_token(
    logits: &mut [f32],
    history: &[usize],
    params: &CausalGenerationParams,
    rng_state: &mut u64,
) -> usize {
    if logits.is_empty() {
        return 0;
    }
    // Apply repetition penalty over seen prompt + generated tokens
    if (params.repetition_penalty - 1.0).abs() > 1e-4 && params.repetition_penalty > 0.0 {
        for &seen in history {
            if let Some(l) = logits.get_mut(seen) {
                if *l > 0.0 {
                    *l /= params.repetition_penalty;
                } else {
                    *l *= params.repetition_penalty;
                }
            }
        }
    }

    // Greedy argmax when temperature <= 1e-5
    if params.temperature <= 1e-5 {
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i;
            }
        }
        return best_id;
    }

    let inv_t = 1.0 / params.temperature.max(1e-4);
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, ((l - max_l) * inv_t).exp()))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    let total_sum: f32 = indexed.iter().map(|x| x.1).sum::<f32>().max(1e-12);
    let cutoff = params.top_p.clamp(0.05, 1.0) * total_sum;
    let mut cum = 0.0f32;
    let mut kept = 0usize;
    for &(_, p) in &indexed {
        cum += p;
        kept += 1;
        if cum >= cutoff {
            break;
        }
    }
    let slice = &indexed[..kept.max(1)];
    let norm_sum: f32 = slice.iter().map(|x| x.1).sum::<f32>().max(1e-12);

    // SplitMix64 PRNG step
    *rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *rng_state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
    let target = u * norm_sum;

    let mut acc = 0.0f32;
    for &(tid, p) in slice {
        acc += p;
        if acc >= target {
            return tid;
        }
    }
    slice[0].0
}

/// Execute full multi-layer causal autoregressive generation (`BpeTokenizer` -> `RoPE` + `GQA` +
/// `RMSNorm` + `SwiGLU` + `LivePrefixKvStore` `Mooncake` KV reuse -> `LMHead` sampling) directly
/// over sub-1-bit [`LittleBitSVFTensor`] bitplanes!
pub fn generate_with_littlebit_causal_decoder(
    harness: &mut JouleHarnessEngine,
    live_kv_store: &mut LivePrefixKvStore,
    model_id: &str,
    prompt: &str,
    params: &CausalGenerationParams,
) -> Result<CausalDecoderReceipt, String> {
    generate_with_littlebit_causal_decoder_accelerated(
        harness,
        live_kv_store,
        model_id,
        prompt,
        params,
        "cpu_tmac_lut_bitplane",
        &mut |_, _, _, _| None,
        &mut |_, _, _, _, _, _, _, _, _| None,
    )
}

/// Hardware-accelerated (`wgpu` Metal/Vulkan resident batched bitplane + fused causal `GQA` hooks,
/// with automatic `T-MAC` 8-bit LUT fallback) multi-layer causal autoregressive decoder!
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn generate_with_littlebit_causal_decoder_accelerated<FProj, FGqa>(
    harness: &mut JouleHarnessEngine,
    live_kv_store: &mut LivePrefixKvStore,
    model_id: &str,
    prompt: &str,
    params: &CausalGenerationParams,
    backend_label: &str,
    batch_proj_hook: &mut FProj,
    causal_gqa_hook: &mut FGqa,
) -> Result<CausalDecoderReceipt, String>
where
    FProj: FnMut(&str, &[f32], usize, &[f32]) -> Option<Vec<f32>>,
    FGqa:
        FnMut(&[f32], &[f32], &[f32], usize, usize, usize, usize, usize, usize) -> Option<Vec<f32>>,
{
    let t0 = Instant::now();
    let (pfx, num_layers) = detect_causal_decoder(harness, model_id).ok_or_else(|| {
        format!(
            "model '{model_id}' does not contain causal/hybrid decoder layers (Llama/Qwen3/LFM2/Bonsai/BitNet/Gemma4/Needle)"
        )
    })?;

    let embed_key = resolve_embed_key(harness, model_id, &pfx)
        .ok_or_else(|| "missing embed_tokens.weight".to_string())?;
    let embed_tok = harness
        .littlebit_tensors
        .get(&embed_key)
        .ok_or_else(|| "missing embed_tokens.weight".to_string())?;
    let hidden_dim = embed_tok.cols;
    let vocab_size = embed_tok.rows;

    let meta = harness.model_configs.get(model_id).cloned();
    let (num_q_heads, num_kv_heads, head_dim) =
        if let Some((q0, k0)) = find_first_attention_q_k(harness, &pfx, num_layers) {
            infer_gqa_geometry(q0.rows, k0.rows, meta.as_ref())
        } else {
            (1, 1, hidden_dim.max(1))
        };
    let rope_theta = meta.as_ref().map(|m| m.rope_theta).unwrap_or(10_000.0);
    let rms_eps = meta.as_ref().map(|m| m.rms_norm_eps).unwrap_or(1e-5);
    let gemma_norm_offset = meta
        .as_ref()
        .map(|m| m.gemma_rms_norm_offset)
        .unwrap_or(false);
    let hidden_act = meta
        .as_ref()
        .map(|m| m.hidden_act.clone())
        .unwrap_or_else(|| "silu".to_string());
    let logit_softcap = meta.as_ref().and_then(|m| m.final_logit_softcapping);

    let mut has_hybrid_or_custom_layers = gemma_norm_offset;
    for l in 0..num_layers {
        let lp = format!("{pfx}layers.{l}");
        if harness
            .littlebit_tensors
            .contains_key(&format!("{lp}.conv.in_proj.weight"))
            || harness
                .littlebit_tensors
                .contains_key(&format!("{lp}.feed_forward.w1.weight"))
            || harness
                .littlebit_tensors
                .contains_key(&format!("{lp}.fc1.weight"))
            || harness
                .vectors_1d
                .contains_key(&format!("{lp}.self_attn.q_norm.weight"))
            || harness
                .vectors_1d
                .contains_key(&format!("{lp}.self_attn.attn_sub_norm.weight"))
        {
            has_hybrid_or_custom_layers = true;
            break;
        }
    }

    let tokenizer = match harness.model_vocabs.get(model_id) {
        Some(v_vec) if !v_vec.is_empty() => {
            let merges = harness
                .model_bpe_merges
                .get(model_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            BpeTokenizer::from_vocab_and_merges(v_vec, merges)
        }
        _ => BpeTokenizer::fallback_for_vocab_size(vocab_size),
    };

    let mut prompt_ids = Vec::new();
    if let Some(bos) = meta.as_ref().and_then(|m| m.bos_token_id)
        && !prompt.starts_with("<|")
        && !prompt.starts_with('<')
    {
        prompt_ids.push(bos.min(vocab_size - 1));
    }
    prompt_ids.extend(tokenizer.encode(prompt, 512));
    if prompt_ids.is_empty() {
        prompt_ids.push(0);
    }

    let mut stop_ids: Vec<usize> = params.stop_token_ids.clone();
    if let Some(ref m) = meta {
        for &eid in &m.eos_token_ids {
            if !stop_ids.contains(&eid) {
                stop_ids.push(eid);
            }
        }
    }
    for stop_str in [
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "</s>",
        "<eos>",
        "<end_of_turn>",
    ] {
        if let Some(&sid) = tokenizer.token_to_id.get(stop_str)
            && !stop_ids.contains(&sid)
        {
            stop_ids.push(sid);
        }
    }

    // Match & populate Mooncake RadixKvCachePool metadata
    let prompt_u32: Vec<u32> = prompt_ids.iter().map(|&x| x as u32).collect();
    let kv_prefix_receipt = harness
        .kv_pool
        .match_and_populate(&prompt_u32, KvStorageTier::UmaSram);

    // Per-layer KV cache & `ShortConv` causal state cache (`LiquidAI LFM2`/`LFM3`)
    let mut k_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); num_layers];
    let mut v_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); num_layers];
    let mut conv_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); num_layers];

    // Reuse materialized `(K, V, ShortConv)` prefix blocks from `live_kv_store`!
    let blk_sz = live_kv_store.block_tokens.max(4);
    let mut matched_live_tokens = 0usize;
    let mut parent_hash = 0u64;
    let mut cached_final_norm: Option<Vec<f32>> = None;

    for chunk in prompt_ids.chunks(blk_sz) {
        if chunk.len() < blk_sz {
            break;
        }
        let h = LivePrefixKvStore::chain_hash(parent_hash, chunk);
        if let Some(entry) = live_kv_store.blocks.get(&(model_id.to_string(), h))
            && entry.tokens == chunk
            && entry.k_layers.len() == num_layers
        {
            for l in 0..num_layers {
                k_cache[l].extend(entry.k_layers[l].iter().cloned());
                v_cache[l].extend(entry.v_layers[l].iter().cloned());
                if let Some(cl) = entry.conv_layers.get(l) {
                    conv_cache[l].extend(cl.iter().cloned());
                }
            }
            matched_live_tokens += chunk.len();
            cached_final_norm = Some(entry.last_hidden_normed.clone());
            parent_hash = h;
        } else {
            break;
        }
    }

    let mut bpw_sum = embed_tok.bits_per_weight();
    let mut bpw_cnt = 1usize;
    for (k, lb) in &harness.littlebit_tensors {
        if k.starts_with(&pfx) {
            bpw_sum += lb.bits_per_weight();
            bpw_cnt += 1;
        }
    }
    let avg_bpw = bpw_sum / (bpw_cnt.max(1) as f64);

    let mut linear_ops = 0usize;
    let mut gpu_dispatches = 0usize;

    let mut run_proj_batch = |tensors: &BTreeMap<String, LittleBitSVFTensor>,
                              vecs_1d: &BTreeMap<String, Vec<f32>>,
                              w_key: &str,
                              b_key: Option<&str>,
                              batch_x: &[f32],
                              batch_sz: usize,
                              ops: &mut usize,
                              g_disp: &mut usize,
                              proj_fn: &mut FProj|
     -> Result<Vec<f32>, String> {
        let t = tensors
            .get(w_key)
            .ok_or_else(|| format!("missing causal tensor '{w_key}'"))?;
        *ops += batch_sz;
        let eff_sigma = t.effective_latent_scales(&[]);
        let mut y = if let Some(mut gpu_out) = proj_fn(w_key, batch_x, batch_sz, &eff_sigma) {
            *g_disp += 1;
            for bi in 0..batch_sz {
                let xb = &batch_x[bi * t.cols..(bi + 1) * t.cols];
                let yb = &mut gpu_out[bi * t.rows..(bi + 1) * t.rows];
                for o in &t.outliers {
                    let r = o.row as usize;
                    let c = o.col as usize;
                    if r < t.rows && c < t.cols {
                        yb[r] += o.delta * xb[c];
                    }
                }
            }
            gpu_out
        } else {
            t.forward_batch(batch_x, batch_sz, &[])?
        };
        if let Some(bk) = b_key
            && let Some(b) = vecs_1d.get(bk)
        {
            for bi in 0..batch_sz {
                let yb = &mut y[bi * t.rows..(bi + 1) * t.rows];
                for (yi, &bv) in yb.iter_mut().zip(b.iter()) {
                    *yi += bv;
                }
            }
        }
        Ok(y)
    };

    // Universal batched forward step over `batch_tids` starting at `start_pos`:
    // natively executes `Llama`/`SmolLM2`/`Qwen3`/`LiquidAI LFM2` (`ShortConv`)/`PrismML Bonsai`/
    // `BitNet b1.58` (`SubLN` + `ReLU²`)/`Google Gemma 4` (`GeGLU` + `+1` RMSNorm)/`Cactus Needle`!
    #[allow(clippy::too_many_arguments)]
    let mut step_token_batch = |batch_tids: &[usize],
                                start_pos: usize,
                                k_cache: &mut [Vec<Vec<f32>>],
                                v_cache: &mut [Vec<Vec<f32>>],
                                conv_cache: &mut [Vec<Vec<f32>>],
                                ops: &mut usize,
                                g_disp: &mut usize,
                                proj_fn: &mut FProj,
                                gqa_fn: &mut FGqa|
     -> Result<Vec<f32>, String> {
        let bsz = batch_tids.len();
        let embed_t = harness
            .littlebit_tensors
            .get(&embed_key)
            .ok_or_else(|| "missing embed_tokens.weight".to_string())?;
        let emb_scale = if gemma_norm_offset {
            (hidden_dim as f32).sqrt()
        } else {
            1.0
        };
        let emb_ln = harness
            .vectors_1d
            .get(&format!("{pfx}embedding_norm.weight"))
            .map(Vec::as_slice);
        let mut h_batch = Vec::with_capacity(bsz * hidden_dim);
        for &tid in batch_tids {
            let mut row = reconstruct_embedding_row(embed_t, tid.min(vocab_size - 1));
            if (emb_scale - 1.0).abs() > 1e-5 {
                for v in &mut row {
                    *v *= emb_scale;
                }
            }
            if emb_ln.is_some() {
                row = rms_norm_cfg(&row, emb_ln, rms_eps, gemma_norm_offset);
            }
            h_batch.extend_from_slice(&row);
        }

        for l in 0..num_layers {
            let lp = format!("{pfx}layers.{l}");
            let in_ln = first_vec_1d(
                &harness.vectors_1d,
                &[
                    format!("{lp}.input_layernorm.weight"),
                    format!("{lp}.operator_norm.weight"),
                    format!("{lp}.self_attn_layer_norm.weight"),
                ],
            );
            let mut h_norm_batch = Vec::with_capacity(bsz * hidden_dim);
            for bi in 0..bsz {
                let sl = &h_batch[bi * hidden_dim..(bi + 1) * hidden_dim];
                h_norm_batch.extend_from_slice(&rms_norm_cfg(
                    sl,
                    in_ln,
                    rms_eps,
                    gemma_norm_offset,
                ));
            }

            // Operator Branch A: Liquid AI `LFM2` / `LFM2.5` / `LFM3` `ShortConv` (`LIV` Gated Depthwise Conv1D)
            if harness
                .littlebit_tensors
                .contains_key(&format!("{lp}.conv.in_proj.weight"))
            {
                let bcx = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.conv.in_proj.weight"),
                    Some(&format!("{lp}.conv.in_proj.bias")),
                    &h_norm_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                let dw = harness
                    .vectors_1d
                    .get(&format!("{lp}.conv.conv.weight"))
                    .map(Vec::as_slice);
                let dw_bias = harness
                    .vectors_1d
                    .get(&format!("{lp}.conv.conv.bias"))
                    .map(Vec::as_slice);
                let k_conv = dw
                    .map(|w| (w.len() / hidden_dim.max(1)).max(1))
                    .unwrap_or_else(|| meta.as_ref().map(|m| m.conv_l_cache).unwrap_or(3));

                let mut gated_batch = vec![0.0f32; bsz * hidden_dim];
                for bi in 0..bsz {
                    let row3 = &bcx[bi * 3 * hidden_dim..(bi + 1) * 3 * hidden_dim];
                    let b_gate = &row3[0..hidden_dim];
                    let c_gate = &row3[hidden_dim..2 * hidden_dim];
                    let x_val = &row3[2 * hidden_dim..3 * hidden_dim];
                    let bx: Vec<f32> = b_gate
                        .iter()
                        .zip(x_val.iter())
                        .map(|(&b, &x)| b * x)
                        .collect();
                    conv_cache[l].push(bx);
                    let hist_len = conv_cache[l].len();
                    let dst = &mut gated_batch[bi * hidden_dim..(bi + 1) * hidden_dim];
                    for ch in 0..hidden_dim {
                        let mut acc = dw_bias.and_then(|b| b.get(ch)).copied().unwrap_or(0.0);
                        for tap in 0..k_conv {
                            // Causal window: tap `k_conv - 1` multiplies newest `hist_len - 1`
                            let back = k_conv - 1 - tap;
                            if hist_len > back {
                                let w_tap = dw
                                    .and_then(|w| w.get(ch * k_conv + tap))
                                    .copied()
                                    .unwrap_or(if tap + 1 == k_conv { 1.0 } else { 0.0 });
                                acc += w_tap * conv_cache[l][hist_len - 1 - back][ch];
                            }
                        }
                        dst[ch] = c_gate[ch] * acc;
                    }
                }
                let conv_out = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.conv.out_proj.weight"),
                    Some(&format!("{lp}.conv.out_proj.bias")),
                    &gated_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                for (hi, &ci) in h_batch.iter_mut().zip(conv_out.iter()) {
                    *hi += ci;
                }
            } else {
                // Operator Branch B: Causal Grouped-Query Self-Attention (`RoPE` + `QK-Norm` + `SubLN`)
                let q_key = format!("{lp}.self_attn.q_proj.weight");
                let k_key = format!("{lp}.self_attn.k_proj.weight");
                let v_key = format!("{lp}.self_attn.v_proj.weight");
                let q_rows = harness
                    .littlebit_tensors
                    .get(&q_key)
                    .map(|t| t.rows)
                    .unwrap_or(num_q_heads * head_dim);
                let kv_rows = harness
                    .littlebit_tensors
                    .get(&k_key)
                    .map(|t| t.rows)
                    .unwrap_or(num_kv_heads * head_dim);
                let (l_qh, l_kvh, l_hd) = infer_gqa_geometry(q_rows, kv_rows, meta.as_ref());
                let l_group = (l_qh / l_kvh.max(1)).max(1);
                let l_inv_sqrt = 1.0 / (l_hd as f32).sqrt();
                let q_dim = l_qh * l_hd;
                let kv_dim = l_kvh * l_hd;

                let mut q_batch = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &q_key,
                    Some(&format!("{lp}.self_attn.q_proj.bias")),
                    &h_norm_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                let mut k_batch = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &k_key,
                    Some(&format!("{lp}.self_attn.k_proj.bias")),
                    &h_norm_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                let v_batch = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &v_key,
                    Some(&format!("{lp}.self_attn.v_proj.bias")),
                    &h_norm_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;

                let q_norm_w = first_vec_1d(
                    &harness.vectors_1d,
                    &[
                        format!("{lp}.self_attn.q_norm.weight"),
                        format!("{lp}.self_attn.q_layernorm.weight"),
                    ],
                );
                let k_norm_w = first_vec_1d(
                    &harness.vectors_1d,
                    &[
                        format!("{lp}.self_attn.k_norm.weight"),
                        format!("{lp}.self_attn.k_layernorm.weight"),
                    ],
                );

                for bi in 0..bsz {
                    let pos = start_pos + bi;
                    let qs = &mut q_batch[bi * q_dim..(bi + 1) * q_dim];
                    let ks = &mut k_batch[bi * kv_dim..(bi + 1) * kv_dim];
                    apply_qk_norm_in_place(qs, l_qh, l_hd, q_norm_w, rms_eps, gemma_norm_offset);
                    apply_qk_norm_in_place(ks, l_kvh, l_hd, k_norm_w, rms_eps, gemma_norm_offset);
                    apply_rope_half_split_in_place(qs, l_qh, l_hd, pos, rope_theta);
                    apply_rope_half_split_in_place(ks, l_kvh, l_hd, pos, rope_theta);
                    k_cache[l].push(ks.to_vec());
                    v_cache[l].push(v_batch[bi * kv_dim..(bi + 1) * kv_dim].to_vec());
                }

                let total_kv = k_cache[l].len();
                let gpu_attn = if bsz >= 4 && total_kv * kv_dim >= 64 {
                    let mut k_flat = Vec::with_capacity(total_kv * kv_dim);
                    let mut v_flat = Vec::with_capacity(total_kv * kv_dim);
                    for s in 0..total_kv {
                        k_flat.extend_from_slice(&k_cache[l][s]);
                        v_flat.extend_from_slice(&v_cache[l][s]);
                    }
                    gqa_fn(
                        &q_batch, &k_flat, &v_flat, bsz, total_kv, start_pos, l_qh, l_kvh, l_hd,
                    )
                } else {
                    None
                };

                let mut attn_batch = if let Some(ga) = gpu_attn {
                    *g_disp += 1;
                    ga
                } else {
                    let mut attn_all = vec![0.0f32; bsz * q_dim];
                    for bi in 0..bsz {
                        let clim = (start_pos + bi + 1).min(total_kv);
                        let q_tok = &q_batch[bi * q_dim..(bi + 1) * q_dim];
                        let out_tok = &mut attn_all[bi * q_dim..(bi + 1) * q_dim];
                        for qh in 0..l_qh {
                            let kvh = (qh / l_group).min(l_kvh.saturating_sub(1));
                            let q_off = qh * l_hd;
                            let kv_off = kvh * l_hd;
                            let q_slice = &q_tok[q_off..q_off + l_hd];

                            let mut scores = Vec::with_capacity(clim);
                            let mut max_s = f32::NEG_INFINITY;
                            for k_step in k_cache[l].iter().take(clim) {
                                let k_slice = &k_step[kv_off..kv_off + l_hd];
                                let dot: f32 =
                                    q_slice.iter().zip(k_slice.iter()).map(|(a, b)| a * b).sum();
                                let sc = dot * l_inv_sqrt;
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
                            let inv_d = 1.0 / denom.max(1e-12);
                            let out_slice = &mut out_tok[q_off..q_off + l_hd];
                            for (s, &p_un) in scores.iter().enumerate() {
                                let p = p_un * inv_d;
                                let v_slice = &v_cache[l][s][kv_off..kv_off + l_hd];
                                for (dst, &vv) in out_slice.iter_mut().zip(v_slice.iter()) {
                                    *dst += p * vv;
                                }
                            }
                        }
                    }
                    attn_all
                };

                // Microsoft `BitNet b1.58` `attn_sub_norm` right before `o_proj`
                if let Some(sub_ln) = harness
                    .vectors_1d
                    .get(&format!("{lp}.self_attn.attn_sub_norm.weight"))
                {
                    for bi in 0..bsz {
                        let sl = &mut attn_batch[bi * q_dim..(bi + 1) * q_dim];
                        let normed = rms_norm_cfg(sl, Some(sub_ln), rms_eps, false);
                        sl.copy_from_slice(&normed);
                    }
                }

                let (o_w_key, o_b_key) = if harness
                    .littlebit_tensors
                    .contains_key(&format!("{lp}.self_attn.o_proj.weight"))
                {
                    (
                        format!("{lp}.self_attn.o_proj.weight"),
                        format!("{lp}.self_attn.o_proj.bias"),
                    )
                } else {
                    (
                        format!("{lp}.self_attn.out_proj.weight"),
                        format!("{lp}.self_attn.out_proj.bias"),
                    )
                };
                let mut o_batch = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &o_w_key,
                    Some(&o_b_key),
                    &attn_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;

                // `Gemma 4` dual-norm topology: when `pre_feedforward_layernorm` exists, `post_attention_layernorm`
                // norms `o_batch` BEFORE adding into residual!
                let has_pre_ffw_ln = harness
                    .vectors_1d
                    .contains_key(&format!("{lp}.pre_feedforward_layernorm.weight"));
                if has_pre_ffw_ln
                    && let Some(post_attn_ln) = harness
                        .vectors_1d
                        .get(&format!("{lp}.post_attention_layernorm.weight"))
                {
                    for bi in 0..bsz {
                        let sl = &mut o_batch[bi * hidden_dim..(bi + 1) * hidden_dim];
                        let normed =
                            rms_norm_cfg(sl, Some(post_attn_ln), rms_eps, gemma_norm_offset);
                        sl.copy_from_slice(&normed);
                    }
                }

                // `Cactus Needle` learned scalar `self_attn_gate`
                if let Some(gate_v) = harness.vectors_1d.get(&format!("{lp}.self_attn_gate"))
                    && let Some(&g0) = gate_v.first()
                {
                    let scale = g0.tanh();
                    for oi in &mut o_batch {
                        *oi *= scale;
                    }
                }

                for (hi, &oi) in h_batch.iter_mut().zip(o_batch.iter()) {
                    *hi += oi;
                }
            }

            // FFN / MLP Block (`SwiGLU`, `LFM2 w1/w3/w2`, `Gemma 4 GeGLU`, `BitNet SubLN + ReLU²`, `Needle fc1/fc2`)
            let ffn_ln = first_vec_1d(
                &harness.vectors_1d,
                &[
                    format!("{lp}.pre_feedforward_layernorm.weight"),
                    format!("{lp}.post_attention_layernorm.weight"),
                    format!("{lp}.ffn_norm.weight"),
                    format!("{lp}.final_layer_norm.weight"),
                ],
            );
            let mut h_ffn_batch = Vec::with_capacity(bsz * hidden_dim);
            for bi in 0..bsz {
                let sl = &h_batch[bi * hidden_dim..(bi + 1) * hidden_dim];
                h_ffn_batch.extend_from_slice(&rms_norm_cfg(
                    sl,
                    ffn_ln,
                    rms_eps,
                    gemma_norm_offset,
                ));
            }

            let mut down_batch = if harness
                .littlebit_tensors
                .contains_key(&format!("{lp}.mlp.gate_proj.weight"))
            {
                let gate_batch = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.mlp.gate_proj.weight"),
                    Some(&format!("{lp}.mlp.gate_proj.bias")),
                    &h_ffn_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                let up_batch = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.mlp.up_proj.weight"),
                    Some(&format!("{lp}.mlp.up_proj.bias")),
                    &h_ffn_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                let ffn_act = if harness
                    .vectors_1d
                    .contains_key(&format!("{lp}.mlp.ffn_sub_norm.weight"))
                    && hidden_act == "silu"
                {
                    "relu2"
                } else {
                    hidden_act.as_str()
                };
                let mut gated: Vec<f32> = gate_batch
                    .iter()
                    .zip(up_batch.iter())
                    .map(|(&g, &u)| apply_gated_act(g, u, ffn_act))
                    .collect();
                if let Some(ffn_sub) = harness
                    .vectors_1d
                    .get(&format!("{lp}.mlp.ffn_sub_norm.weight"))
                {
                    let inter_dim = gated.len() / bsz.max(1);
                    for bi in 0..bsz {
                        let sl = &mut gated[bi * inter_dim..(bi + 1) * inter_dim];
                        let normed = rms_norm_cfg(sl, Some(ffn_sub), rms_eps, false);
                        sl.copy_from_slice(&normed);
                    }
                }
                run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.mlp.down_proj.weight"),
                    Some(&format!("{lp}.mlp.down_proj.bias")),
                    &gated,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?
            } else if harness
                .littlebit_tensors
                .contains_key(&format!("{lp}.feed_forward.w1.weight"))
            {
                // Liquid AI `LFM2` / `LFM2.5` / `LFM3` `SwiGLU` (`w2(silu(w1(x)) * w3(x))`)
                let w1_b = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.feed_forward.w1.weight"),
                    None,
                    &h_ffn_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                let w3_b = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.feed_forward.w3.weight"),
                    None,
                    &h_ffn_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                let gated: Vec<f32> = w1_b
                    .iter()
                    .zip(w3_b.iter())
                    .map(|(&g, &u)| silu(g) * u)
                    .collect();
                run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.feed_forward.w2.weight"),
                    None,
                    &gated,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?
            } else if harness
                .littlebit_tensors
                .contains_key(&format!("{lp}.fc1.weight"))
            {
                // `Cactus Needle` 2-layer MLP (`fc1 -> gelu -> fc2`)
                let mut mid = run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.fc1.weight"),
                    Some(&format!("{lp}.fc1.bias")),
                    &h_ffn_batch,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?;
                for v in &mut mid {
                    *v = gelu_tanh(*v);
                }
                run_proj_batch(
                    &harness.littlebit_tensors,
                    &harness.vectors_1d,
                    &format!("{lp}.fc2.weight"),
                    Some(&format!("{lp}.fc2.bias")),
                    &mid,
                    bsz,
                    ops,
                    g_disp,
                    proj_fn,
                )?
            } else {
                vec![0.0f32; bsz * hidden_dim]
            };

            if let Some(post_ffw_ln) = harness
                .vectors_1d
                .get(&format!("{lp}.post_feedforward_layernorm.weight"))
            {
                for bi in 0..bsz {
                    let sl = &mut down_batch[bi * hidden_dim..(bi + 1) * hidden_dim];
                    let normed = rms_norm_cfg(sl, Some(post_ffw_ln), rms_eps, gemma_norm_offset);
                    sl.copy_from_slice(&normed);
                }
            }
            for (hi, &di) in h_batch.iter_mut().zip(down_batch.iter()) {
                *hi += di;
            }
        }

        let final_ln = first_vec_1d(
            &harness.vectors_1d,
            &[
                format!("{pfx}norm.weight"),
                format!("{pfx}layer_norm.weight"),
                format!("{model_id}::model.norm.weight"),
            ],
        );
        let mut normed_batch = Vec::with_capacity(bsz * hidden_dim);
        for bi in 0..bsz {
            let sl = &h_batch[bi * hidden_dim..(bi + 1) * hidden_dim];
            normed_batch.extend_from_slice(&rms_norm_cfg(sl, final_ln, rms_eps, gemma_norm_offset));
        }
        Ok(normed_batch)
    };

    // 1. Batched prefill over ALL uncached suffix tokens of `prompt_ids` & populate `live_kv_store`!
    let mut last_normed = cached_final_norm.unwrap_or_else(|| vec![0.0f32; hidden_dim]);
    let mut block_parent_h = 0u64;
    for b_idx in 0..(matched_live_tokens / blk_sz) {
        let chunk = &prompt_ids[b_idx * blk_sz..(b_idx + 1) * blk_sz];
        block_parent_h = LivePrefixKvStore::chain_hash(block_parent_h, chunk);
    }

    if matched_live_tokens < prompt_ids.len() {
        let uncached_slice = &prompt_ids[matched_live_tokens..];
        let normed_prefill = step_token_batch(
            uncached_slice,
            matched_live_tokens,
            &mut k_cache,
            &mut v_cache,
            &mut conv_cache,
            &mut linear_ops,
            &mut gpu_dispatches,
            batch_proj_hook,
            causal_gqa_hook,
        )?;
        for (idx, pos) in (matched_live_tokens..prompt_ids.len()).enumerate() {
            let tok_normed = &normed_prefill[idx * hidden_dim..(idx + 1) * hidden_dim];
            if pos + 1 == prompt_ids.len() {
                last_normed = tok_normed.to_vec();
            }
            if (pos + 1).is_multiple_of(blk_sz) {
                let blk_start = pos + 1 - blk_sz;
                let chunk = &prompt_ids[blk_start..pos + 1];
                let h = LivePrefixKvStore::chain_hash(block_parent_h, chunk);
                let key = (model_id.to_string(), h);
                if !live_kv_store.blocks.contains_key(&key) {
                    if live_kv_store.blocks.len() >= MAX_LIVE_KV_BLOCKS
                        && let Some(old_k) = live_kv_store.fifo.pop_front()
                    {
                        live_kv_store.blocks.remove(&old_k);
                    }
                    let mut k_layers = Vec::with_capacity(num_layers);
                    let mut v_layers = Vec::with_capacity(num_layers);
                    let mut c_layers = Vec::with_capacity(num_layers);
                    for l in 0..num_layers {
                        k_layers.push(if k_cache[l].len() >= pos + 1 {
                            k_cache[l][blk_start..pos + 1].to_vec()
                        } else {
                            Vec::new()
                        });
                        v_layers.push(if v_cache[l].len() >= pos + 1 {
                            v_cache[l][blk_start..pos + 1].to_vec()
                        } else {
                            Vec::new()
                        });
                        c_layers.push(if conv_cache[l].len() >= pos + 1 {
                            conv_cache[l][blk_start..pos + 1].to_vec()
                        } else {
                            Vec::new()
                        });
                    }
                    live_kv_store.fifo.push_back(key.clone());
                    live_kv_store.blocks.insert(
                        key,
                        CachedKvBlockTensors {
                            tokens: chunk.to_vec(),
                            k_layers,
                            v_layers,
                            conv_layers: c_layers,
                            last_hidden_normed: tok_normed.to_vec(),
                        },
                    );
                }
                block_parent_h = h;
            }
        }
    }

    // Resolve `lm_head` (explicit `lm_head.weight` or weight-tied `embed_tokens.weight`)
    let lm_head_key = [
        format!("{model_id}::lm_head.weight"),
        format!("{pfx}lm_head.weight"),
        embed_key.clone(),
    ]
    .into_iter()
    .find(|k| harness.littlebit_tensors.contains_key(k))
    .unwrap_or_else(|| embed_key.clone());
    let logit_vocab_size = harness
        .littlebit_tensors
        .get(&lm_head_key)
        .map(|t| t.rows)
        .unwrap_or(vocab_size);

    let apply_softcap = |logits: &mut [f32]| {
        if let Some(cap) = logit_softcap
            && cap > 0.0
        {
            for l in logits.iter_mut() {
                *l = cap * (*l / cap).tanh();
            }
        }
    };

    // 2. Autoregressive generation loop (supports lossless Self-Speculative `PLD` + Rank-8 Draft-and-Verify!)
    let max_new = params.max_new_tokens.clamp(1, 512);
    let mut history = prompt_ids.clone();
    let mut generated_ids = Vec::with_capacity(max_new);
    let mut rng_state = params.seed ^ 0xA5A5_5A5A_F0F0_0F0F;
    let mut spec_drafted = 0usize;
    let mut spec_accepted = 0usize;

    // Sample the very first completion token directly from prefill `last_normed`
    let mut first_logits = run_proj_batch(
        &harness.littlebit_tensors,
        &harness.vectors_1d,
        &lm_head_key,
        None,
        &last_normed,
        1,
        &mut linear_ops,
        &mut gpu_dispatches,
        batch_proj_hook,
    )?;
    apply_softcap(&mut first_logits);
    let first_id = sample_next_token(&mut first_logits, &history, params, &mut rng_state);
    generated_ids.push(first_id);
    history.push(first_id);

    let spec_k = params.speculative_draft_k.min(8);
    let kv_group_size = (num_q_heads / num_kv_heads.max(1)).max(1);
    let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

    if spec_k == 0 || max_new <= 1 {
        let mut prev_id = first_id;
        for step in 1..max_new {
            let next_pos = history.len() - 1;
            last_normed = step_token_batch(
                &[prev_id],
                next_pos,
                &mut k_cache,
                &mut v_cache,
                &mut conv_cache,
                &mut linear_ops,
                &mut gpu_dispatches,
                batch_proj_hook,
                causal_gqa_hook,
            )?;
            let mut logits = run_proj_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &lm_head_key,
                None,
                &last_normed,
                1,
                &mut linear_ops,
                &mut gpu_dispatches,
                batch_proj_hook,
            )?;
            apply_softcap(&mut logits);
            let next_id = sample_next_token(&mut logits, &history, params, &mut rng_state);
            if stop_ids.contains(&next_id) {
                break;
            }
            generated_ids.push(next_id);
            history.push(next_id);
            prev_id = next_id;
            let _ = step;
        }
    } else {
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut cur_tok = first_id;
        let mut hit_stop = false;
        let mut neural_drafted = 0usize;
        let mut neural_accepted = 0usize;
        let mut pld_drafted = 0usize;
        let mut pld_accepted = 0usize;

        let lookup_pld = |hist: &[usize], max_k: usize| -> Vec<usize> {
            if max_k == 0 || hist.len() < 5 {
                return Vec::new();
            }
            for ng in [4usize, 3] {
                if hist.len() <= ng + 1 {
                    continue;
                }
                let needle = &hist[hist.len() - ng..];
                let search_end = hist.len() - ng - 1;
                for pos in (0..search_end).rev() {
                    if &hist[pos..pos + ng] == needle {
                        let mut out = Vec::with_capacity(max_k);
                        for &cand in hist[pos + ng..hist.len() - 1].iter().take(max_k) {
                            if stop_ids.contains(&cand) {
                                break;
                            }
                            out.push(cand);
                        }
                        if !out.is_empty() {
                            return out;
                        }
                    }
                }
            }
            Vec::new()
        };

        while generated_ids.len() < max_new && !hit_stop {
            let rem = max_new - generated_ids.len();
            let k_draft = spec_k.min(rem.saturating_sub(1));
            let pos_0 = history.len() - 1;

            // 2a. Fast Two-Tier Self-Speculative Rollout (`PLD` on all models; Nested Half-Rank on pure GQA/SwiGLU)
            let mut draft_proposals = Vec::with_capacity(k_draft);
            let mut step_used_neural = false;
            let mut step_used_pld = false;
            if k_draft > 0 {
                let pld_ok = pld_drafted < 8 || (pld_accepted * 100 >= pld_drafted * 35);
                let pld_cands = if pld_ok {
                    lookup_pld(&history, k_draft)
                } else {
                    Vec::new()
                };
                if !pld_cands.is_empty() {
                    step_used_pld = true;
                    pld_drafted += pld_cands.len();
                    draft_proposals.extend_from_slice(&pld_cands);
                } else if !has_hybrid_or_custom_layers {
                    let neural_ok =
                        neural_drafted < 6 || (neural_accepted * 100 >= neural_drafted * 40);
                    if neural_ok {
                        step_used_neural = true;
                        let k_nn = k_draft.min(2);
                        let mut temp_k: Vec<Vec<Vec<f32>>> = k_cache.to_vec();
                        let mut temp_v: Vec<Vec<Vec<f32>>> = v_cache.to_vec();
                        let mut draft_hist = history.clone();
                        let mut draft_rng = rng_state;
                        let mut d_tok = cur_tok;

                        for d_step in 0..k_nn {
                            let d_pos = pos_0 + d_step;
                            let embed_t = harness
                                .littlebit_tensors
                                .get(&embed_key)
                                .ok_or_else(|| "missing embed_tokens.weight".to_string())?;
                            let mut h = reconstruct_embedding_row(embed_t, d_tok);

                            for l in 0..num_layers {
                                let lp = format!("{pfx}layers.{l}");
                                let in_ln = harness
                                    .vectors_1d
                                    .get(&format!("{lp}.input_layernorm.weight"))
                                    .map(Vec::as_slice);
                                let h_norm = rms_norm(&h, in_ln, rms_eps);

                                let q_t = harness
                                    .littlebit_tensors
                                    .get(&format!("{lp}.self_attn.q_proj.weight"))
                                    .ok_or_else(|| "missing q_proj".to_string())?;
                                let k_t = harness
                                    .littlebit_tensors
                                    .get(&format!("{lp}.self_attn.k_proj.weight"))
                                    .ok_or_else(|| "missing k_proj".to_string())?;
                                let v_t = harness
                                    .littlebit_tensors
                                    .get(&format!("{lp}.self_attn.v_proj.weight"))
                                    .ok_or_else(|| "missing v_proj".to_string())?;
                                let o_t = harness
                                    .littlebit_tensors
                                    .get(&format!("{lp}.self_attn.o_proj.weight"))
                                    .ok_or_else(|| "missing o_proj".to_string())?;

                                let r_q = (q_t.rank / 2).max(4);
                                let r_k = (k_t.rank / 2).max(4);
                                let r_v = (v_t.rank / 2).max(4);
                                let r_o = (o_t.rank / 2).max(4);

                                let mut q = q_t.forward_vec_truncated_rank(&h_norm, r_q)?;
                                if let Some(b) = harness
                                    .vectors_1d
                                    .get(&format!("{lp}.self_attn.q_proj.bias"))
                                {
                                    for (qi, &bi) in q.iter_mut().zip(b.iter()) {
                                        *qi += bi;
                                    }
                                }
                                let mut k = k_t.forward_vec_truncated_rank(&h_norm, r_k)?;
                                if let Some(b) = harness
                                    .vectors_1d
                                    .get(&format!("{lp}.self_attn.k_proj.bias"))
                                {
                                    for (ki, &bi) in k.iter_mut().zip(b.iter()) {
                                        *ki += bi;
                                    }
                                }
                                let mut v = v_t.forward_vec_truncated_rank(&h_norm, r_v)?;
                                if let Some(b) = harness
                                    .vectors_1d
                                    .get(&format!("{lp}.self_attn.v_proj.bias"))
                                {
                                    for (vi, &bi) in v.iter_mut().zip(b.iter()) {
                                        *vi += bi;
                                    }
                                }
                                apply_rope_half_split_in_place(
                                    &mut q,
                                    num_q_heads,
                                    head_dim,
                                    d_pos,
                                    rope_theta,
                                );
                                apply_rope_half_split_in_place(
                                    &mut k,
                                    num_kv_heads,
                                    head_dim,
                                    d_pos,
                                    rope_theta,
                                );
                                temp_k[l].push(k);
                                temp_v[l].push(v);

                                let seq_len = temp_k[l].len();
                                let mut attn_out = vec![0.0f32; q_dim];
                                for qh in 0..num_q_heads {
                                    let kvh =
                                        (qh / kv_group_size).min(num_kv_heads.saturating_sub(1));
                                    let q_sl = &q[qh * head_dim..(qh + 1) * head_dim];
                                    let kv_off = kvh * head_dim;
                                    let mut scores = Vec::with_capacity(seq_len);
                                    let mut mx = f32::NEG_INFINITY;
                                    for ks in &temp_k[l] {
                                        let k_sl = &ks[kv_off..kv_off + head_dim];
                                        let dot: f32 =
                                            q_sl.iter().zip(k_sl.iter()).map(|(a, b)| a * b).sum();
                                        let sc = dot * inv_sqrt_hd;
                                        if sc > mx {
                                            mx = sc;
                                        }
                                        scores.push(sc);
                                    }
                                    let mut den = 0.0f32;
                                    for sc in &mut scores {
                                        *sc = (*sc - mx).exp();
                                        den += *sc;
                                    }
                                    let inv_d = 1.0 / den.max(1e-12);
                                    let dst_sl = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];
                                    for (s_i, &w) in scores.iter().enumerate() {
                                        let p = w * inv_d;
                                        let v_sl = &temp_v[l][s_i][kv_off..kv_off + head_dim];
                                        for (dst, &vv) in dst_sl.iter_mut().zip(v_sl.iter()) {
                                            *dst += p * vv;
                                        }
                                    }
                                }
                                let mut o_vec = o_t.forward_vec_truncated_rank(&attn_out, r_o)?;
                                if let Some(b) = harness
                                    .vectors_1d
                                    .get(&format!("{lp}.self_attn.o_proj.bias"))
                                {
                                    for (oi, &bi) in o_vec.iter_mut().zip(b.iter()) {
                                        *oi += bi;
                                    }
                                }
                                for (hi, &oi) in h.iter_mut().zip(o_vec.iter()) {
                                    *hi += oi;
                                }

                                let post_ln = harness
                                    .vectors_1d
                                    .get(&format!("{lp}.post_attention_layernorm.weight"))
                                    .map(Vec::as_slice);
                                let h_ffn = rms_norm(&h, post_ln, rms_eps);
                                let gate_t = harness
                                    .littlebit_tensors
                                    .get(&format!("{lp}.mlp.gate_proj.weight"))
                                    .ok_or_else(|| "missing gate_proj".to_string())?;
                                let up_t = harness
                                    .littlebit_tensors
                                    .get(&format!("{lp}.mlp.up_proj.weight"))
                                    .ok_or_else(|| "missing up_proj".to_string())?;
                                let down_t = harness
                                    .littlebit_tensors
                                    .get(&format!("{lp}.mlp.down_proj.weight"))
                                    .ok_or_else(|| "missing down_proj".to_string())?;
                                let gate = gate_t
                                    .forward_vec_truncated_rank(&h_ffn, (gate_t.rank / 2).max(4))?;
                                let up = up_t
                                    .forward_vec_truncated_rank(&h_ffn, (up_t.rank / 2).max(4))?;
                                let swiglu: Vec<f32> = gate
                                    .iter()
                                    .zip(up.iter())
                                    .map(|(&g, &u)| silu(g) * u)
                                    .collect();
                                let down = down_t.forward_vec_truncated_rank(
                                    &swiglu,
                                    (down_t.rank / 2).max(4),
                                )?;
                                for (hi, &di) in h.iter_mut().zip(down.iter()) {
                                    *hi += di;
                                }
                                let _ = kv_dim;
                            }

                            let final_ln = harness
                                .vectors_1d
                                .get(&format!("{pfx}norm.weight"))
                                .map(Vec::as_slice);
                            let d_normed = rms_norm(&h, final_ln, rms_eps);
                            let lm_t = harness
                                .littlebit_tensors
                                .get(&lm_head_key)
                                .ok_or_else(|| "missing lm_head".to_string())?;
                            let mut d_logits =
                                lm_t.forward_vec_truncated_rank(&d_normed, (lm_t.rank / 2).max(4))?;
                            let cand = sample_next_token(
                                &mut d_logits,
                                &draft_hist,
                                params,
                                &mut draft_rng,
                            );
                            if stop_ids.contains(&cand) {
                                break;
                            }
                            draft_proposals.push(cand);
                            draft_hist.push(cand);
                            d_tok = cand;
                        }
                        neural_drafted += draft_proposals.len();
                    }
                }
                spec_drafted += draft_proposals.len();
            }

            // 2b. Parallel Batched Full-Model Target Verification (`batch_sz = 1 + draft_proposals.len()`)
            let mut verify_tokens = Vec::with_capacity(1 + draft_proposals.len());
            verify_tokens.push(cur_tok);
            verify_tokens.extend_from_slice(&draft_proposals);
            let b_verify = verify_tokens.len();

            let normed_verify = step_token_batch(
                &verify_tokens,
                pos_0,
                &mut k_cache,
                &mut v_cache,
                &mut conv_cache,
                &mut linear_ops,
                &mut gpu_dispatches,
                batch_proj_hook,
                causal_gqa_hook,
            )?;
            let mut logits_verify = run_proj_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &lm_head_key,
                None,
                &normed_verify,
                b_verify,
                &mut linear_ops,
                &mut gpu_dispatches,
                batch_proj_hook,
            )?;
            apply_softcap(&mut logits_verify);

            for d in 0..b_verify {
                let l_slice = &mut logits_verify[d * logit_vocab_size..(d + 1) * logit_vocab_size];
                let target_next = sample_next_token(l_slice, &history, params, &mut rng_state);
                if d < draft_proposals.len()
                    && target_next == draft_proposals[d]
                    && !stop_ids.contains(&target_next)
                {
                    generated_ids.push(target_next);
                    history.push(target_next);
                    spec_accepted += 1;
                    if step_used_neural {
                        neural_accepted += 1;
                    }
                    if step_used_pld {
                        pld_accepted += 1;
                    }
                } else {
                    let keep_len = pos_0 + d + 1;
                    for l in 0..num_layers {
                        if k_cache[l].len() > keep_len {
                            k_cache[l].truncate(keep_len);
                            v_cache[l].truncate(keep_len);
                        }
                        if conv_cache[l].len() > keep_len {
                            conv_cache[l].truncate(keep_len);
                        }
                    }
                    if stop_ids.contains(&target_next) {
                        hit_stop = true;
                    } else if generated_ids.len() < max_new {
                        generated_ids.push(target_next);
                        history.push(target_next);
                        cur_tok = target_next;
                    }
                    break;
                }
            }
        }
    }

    let generated_text = tokenizer.decode(&generated_ids, true);
    let elapsed_ns = (t0.elapsed().as_nanos() as u64).max(1);
    let tokens_per_sec = (generated_ids.len() as f64) / ((elapsed_ns as f64) * 1e-9);
    let spec_rate = if spec_drafted > 0 {
        (spec_accepted as f64) / (spec_drafted as f64)
    } else {
        0.0
    };

    Ok(CausalDecoderReceipt {
        model_id: model_id.to_string(),
        num_layers,
        num_q_heads,
        num_kv_heads,
        head_dim,
        hidden_dim,
        prompt_token_ids: prompt_ids,
        generated_token_ids: generated_ids,
        generated_text,
        avg_bits_per_weight: avg_bpw,
        elapsed_ns,
        tokens_per_sec,
        linear_projections_executed: linear_ops,
        kv_prefix_receipt,
        gpu_accelerated: gpu_dispatches > 0,
        gpu_dispatches,
        execution_backend: backend_label.to_string(),
        speculative_drafted_tokens: spec_drafted,
        speculative_accepted_tokens: spec_accepted,
        speculative_acceptance_rate: spec_rate,
    })
}

/// Per-request completion receipt inside a coalesced multi-request continuous batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedCausalItemReceipt {
    pub request_index: usize,
    pub prompt: String,
    pub generated_token_ids: Vec<usize>,
    pub generated_text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Aggregate receipt from Continuous In-Flight Multi-Request Batched Decoding (`Orca`/`vLLM`-class
/// lockstep weight-sharing over contiguous row-outer `T-MAC` LUT bitplanes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedCausalDecoderReceipt {
    pub model_id: String,
    pub batch_size: usize,
    pub num_layers: usize,
    pub total_prompt_tokens: usize,
    pub total_generated_tokens: usize,
    pub avg_bits_per_weight: f64,
    pub elapsed_ns: u64,
    pub aggregate_tokens_per_sec: f64,
    pub shared_batch_projections: usize,
    pub items: Vec<BatchedCausalItemReceipt>,
}

/// Execute Continuous In-Flight Multi-Request Batched Generation over `prompts`:
/// coalesces all active requests at every decode step into a single row-outer `T-MAC` LUT
/// `forward_batch` pass per weight matrix so `u_bits` bitplanes are loaded from memory ONCE
/// across all concurrent tenants!
pub fn generate_batch_with_littlebit_causal_decoder(
    harness: &mut JouleHarnessEngine,
    live_kv: &mut LivePrefixKvStore,
    model_id: &str,
    prompts: &[String],
    params: &CausalGenerationParams,
) -> Result<BatchedCausalDecoderReceipt, String> {
    if prompts.is_empty() {
        return Err("prompts batch must contain at least 1 prompt".to_string());
    }
    let t0 = Instant::now();
    let (pfx, num_layers) = detect_causal_decoder(harness, model_id)
        .ok_or_else(|| format!("model '{model_id}' is not a causal decoder"))?;

    let embed_key = resolve_embed_key(harness, model_id, &pfx)
        .ok_or_else(|| "missing embed_tokens".to_string())?;
    let embed_tok = harness
        .littlebit_tensors
        .get(&embed_key)
        .ok_or_else(|| "missing embed_tokens".to_string())?;
    let hidden_dim = embed_tok.cols;
    let lm_head_key = [
        format!("{model_id}::lm_head.weight"),
        format!("{pfx}lm_head.weight"),
        embed_key.clone(),
    ]
    .into_iter()
    .find(|k| harness.littlebit_tensors.contains_key(k))
    .ok_or_else(|| "missing lm_head / embed_tokens".to_string())?;

    let cfg = harness.model_configs.get(model_id).cloned();
    let (num_q_heads, num_kv_heads, head_dim) =
        if let Some((q0, k0)) = find_first_attention_q_k(harness, &pfx, num_layers) {
            infer_gqa_geometry(q0.rows, k0.rows, cfg.as_ref())
        } else {
            (1, 1, hidden_dim.max(1))
        };
    let vocab_size = harness
        .littlebit_tensors
        .get(&lm_head_key)
        .map(|t| t.rows)
        .unwrap_or(embed_tok.rows);

    let rope_theta = cfg.as_ref().map(|c| c.rope_theta).unwrap_or(10_000.0);
    let rms_eps = cfg.as_ref().map(|c| c.rms_norm_eps).unwrap_or(1e-5);
    let gemma_norm_offset = cfg
        .as_ref()
        .map(|c| c.gemma_rms_norm_offset)
        .unwrap_or(false);
    let hidden_act = cfg
        .as_ref()
        .map(|c| c.hidden_act.clone())
        .unwrap_or_else(|| "silu".to_string());
    let mut stop_ids: Vec<usize> = params.stop_token_ids.clone();
    if let Some(ref c) = cfg {
        for &eid in &c.eos_token_ids {
            if !stop_ids.contains(&eid) {
                stop_ids.push(eid);
            }
        }
    }

    let tokenizer = if let Some(vocab) = harness.model_vocabs.get(model_id) {
        let merges = harness
            .model_bpe_merges
            .get(model_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        BpeTokenizer::from_vocab_and_merges(vocab, merges)
    } else {
        BpeTokenizer::fallback_for_vocab_size(vocab_size)
    };

    let model_pfx = format!("{model_id}::");
    let mut bpw_sum = 0.0;
    let mut bpw_cnt = 0usize;
    for (k, lb) in &harness.littlebit_tensors {
        if k.starts_with(&model_pfx) {
            bpw_sum += lb.bits_per_weight();
            bpw_cnt += 1;
        }
    }
    let avg_bpw = bpw_sum / (bpw_cnt.max(1) as f64);

    let active_mix: Vec<(String, f32)> = Vec::new();

    let max_new = params.max_new_tokens.clamp(1, 256);
    let bsz_total = prompts.len();

    let mut req_k_caches: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![Vec::new(); num_layers]; bsz_total];
    let mut req_v_caches: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![Vec::new(); num_layers]; bsz_total];
    let mut req_conv_caches: Vec<Vec<Vec<Vec<f32>>>> =
        vec![vec![Vec::new(); num_layers]; bsz_total];
    let mut req_histories: Vec<Vec<usize>> = Vec::with_capacity(bsz_total);
    let mut req_gen_ids: Vec<Vec<usize>> = vec![Vec::with_capacity(max_new); bsz_total];
    let mut req_prompt_lens: Vec<usize> = Vec::with_capacity(bsz_total);
    let mut req_cur_tok: Vec<usize> = vec![0usize; bsz_total];
    let mut req_done: Vec<bool> = vec![false; bsz_total];
    let mut req_rng: Vec<u64> = (0..bsz_total)
        .map(|i| {
            params
                .seed
                .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .max(1)
        })
        .collect();
    let mut shared_ops = 0usize;

    let run_lb_batch = |tensors: &BTreeMap<String, LittleBitSVFTensor>,
                        vecs_1d: &BTreeMap<String, Vec<f32>>,
                        w_key: &str,
                        b_key: Option<&str>,
                        x_in: &[f32],
                        bsz: usize,
                        ops: &mut usize|
     -> Result<Vec<f32>, String> {
        let t = tensors
            .get(w_key)
            .ok_or_else(|| format!("missing tensor '{w_key}'"))?;
        *ops += 1;
        let mut y = t.forward_batch(x_in, bsz, &active_mix)?;
        if let Some(bk) = b_key
            && let Some(bv) = vecs_1d.get(bk)
        {
            for bi in 0..bsz {
                let sl = &mut y[bi * t.rows..(bi + 1) * t.rows];
                for (dst, &b) in sl.iter_mut().zip(bv.iter()) {
                    *dst += b;
                }
            }
        }
        Ok(y)
    };

    let kv_group_size = (num_q_heads / num_kv_heads.max(1)).max(1);
    let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let _ = live_kv;

    // Helper for running 1 layer across a slice of request indices (either `plen` tokens of a single
    // request during prefill or `b_act` single tokens across active requests during lockstep decode!)
    let run_multi_req_layer = |l: usize,
                               h_buf: &mut [f32],
                               bsz: usize,
                               req_map: &[(usize, usize)], // (req_idx, pos)
                               k_caches: &mut [Vec<Vec<Vec<f32>>>],
                               v_caches: &mut [Vec<Vec<Vec<f32>>>],
                               c_caches: &mut [Vec<Vec<Vec<f32>>>],
                               ops: &mut usize|
     -> Result<(), String> {
        let lp = format!("{pfx}layers.{l}");
        let in_ln = first_vec_1d(
            &harness.vectors_1d,
            &[
                format!("{lp}.input_layernorm.weight"),
                format!("{lp}.operator_norm.weight"),
                format!("{lp}.self_attn_layer_norm.weight"),
            ],
        );
        let mut hn = Vec::with_capacity(bsz * hidden_dim);
        for bi in 0..bsz {
            hn.extend_from_slice(&rms_norm_cfg(
                &h_buf[bi * hidden_dim..(bi + 1) * hidden_dim],
                in_ln,
                rms_eps,
                gemma_norm_offset,
            ));
        }

        if harness
            .littlebit_tensors
            .contains_key(&format!("{lp}.conv.in_proj.weight"))
        {
            let bcx = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.conv.in_proj.weight"),
                Some(&format!("{lp}.conv.in_proj.bias")),
                &hn,
                bsz,
                ops,
            )?;
            let dw = harness
                .vectors_1d
                .get(&format!("{lp}.conv.conv.weight"))
                .map(Vec::as_slice);
            let k_conv = dw
                .map(|w| (w.len() / hidden_dim.max(1)).max(1))
                .unwrap_or(3);
            let mut gated_b = vec![0.0f32; bsz * hidden_dim];
            for (bi, &(ri, _)) in req_map.iter().enumerate() {
                let row3 = &bcx[bi * 3 * hidden_dim..(bi + 1) * 3 * hidden_dim];
                let bx: Vec<f32> = row3[0..hidden_dim]
                    .iter()
                    .zip(row3[2 * hidden_dim..3 * hidden_dim].iter())
                    .map(|(&b, &x)| b * x)
                    .collect();
                c_caches[ri][l].push(bx);
                let hlen = c_caches[ri][l].len();
                let dst = &mut gated_b[bi * hidden_dim..(bi + 1) * hidden_dim];
                for ch in 0..hidden_dim {
                    let mut acc = 0.0f32;
                    for tap in 0..k_conv {
                        let back = k_conv - 1 - tap;
                        if hlen > back {
                            let w_tap = dw
                                .and_then(|w| w.get(ch * k_conv + tap))
                                .copied()
                                .unwrap_or(if tap + 1 == k_conv { 1.0 } else { 0.0 });
                            acc += w_tap * c_caches[ri][l][hlen - 1 - back][ch];
                        }
                    }
                    dst[ch] = row3[hidden_dim + ch] * acc;
                }
            }
            let cout = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.conv.out_proj.weight"),
                Some(&format!("{lp}.conv.out_proj.bias")),
                &gated_b,
                bsz,
                ops,
            )?;
            for (hi, &ci) in h_buf.iter_mut().zip(cout.iter()) {
                *hi += ci;
            }
        } else {
            let mut qb = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.self_attn.q_proj.weight"),
                Some(&format!("{lp}.self_attn.q_proj.bias")),
                &hn,
                bsz,
                ops,
            )?;
            let mut kb = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.self_attn.k_proj.weight"),
                Some(&format!("{lp}.self_attn.k_proj.bias")),
                &hn,
                bsz,
                ops,
            )?;
            let vb = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.self_attn.v_proj.weight"),
                Some(&format!("{lp}.self_attn.v_proj.bias")),
                &hn,
                bsz,
                ops,
            )?;
            let q_norm_w = first_vec_1d(
                &harness.vectors_1d,
                &[
                    format!("{lp}.self_attn.q_norm.weight"),
                    format!("{lp}.self_attn.q_layernorm.weight"),
                ],
            );
            let k_norm_w = first_vec_1d(
                &harness.vectors_1d,
                &[
                    format!("{lp}.self_attn.k_norm.weight"),
                    format!("{lp}.self_attn.k_layernorm.weight"),
                ],
            );
            let mut attn_all = vec![0.0f32; bsz * q_dim];
            for (bi, &(ri, pos)) in req_map.iter().enumerate() {
                let qs = &mut qb[bi * q_dim..(bi + 1) * q_dim];
                let ks = &mut kb[bi * kv_dim..(bi + 1) * kv_dim];
                apply_qk_norm_in_place(
                    qs,
                    num_q_heads,
                    head_dim,
                    q_norm_w,
                    rms_eps,
                    gemma_norm_offset,
                );
                apply_qk_norm_in_place(
                    ks,
                    num_kv_heads,
                    head_dim,
                    k_norm_w,
                    rms_eps,
                    gemma_norm_offset,
                );
                apply_rope_half_split_in_place(qs, num_q_heads, head_dim, pos, rope_theta);
                apply_rope_half_split_in_place(ks, num_kv_heads, head_dim, pos, rope_theta);
                k_caches[ri][l].push(ks.to_vec());
                v_caches[ri][l].push(vb[bi * kv_dim..(bi + 1) * kv_dim].to_vec());

                let clim = k_caches[ri][l].len();
                let ot = &mut attn_all[bi * q_dim..(bi + 1) * q_dim];
                for qh in 0..num_q_heads {
                    let kvh = (qh / kv_group_size).min(num_kv_heads - 1);
                    let qo = qh * head_dim;
                    let kvo = kvh * head_dim;
                    let qsl = &qs[qo..qo + head_dim];
                    let mut scs = Vec::with_capacity(clim);
                    let mut mx = f32::NEG_INFINITY;
                    for k_step in k_caches[ri][l].iter().take(clim) {
                        let dot: f32 = qsl
                            .iter()
                            .zip(k_step[kvo..kvo + head_dim].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        let sc = dot * inv_sqrt_hd;
                        if sc > mx {
                            mx = sc;
                        }
                        scs.push(sc);
                    }
                    let mut den = 0.0f32;
                    for sc in &mut scs {
                        *sc = (*sc - mx).exp();
                        den += *sc;
                    }
                    let inv_d = 1.0 / den.max(1e-12);
                    let dst = &mut ot[qo..qo + head_dim];
                    for (s_i, &w) in scs.iter().enumerate() {
                        let p = w * inv_d;
                        for (d, &vv) in dst
                            .iter_mut()
                            .zip(v_caches[ri][l][s_i][kvo..kvo + head_dim].iter())
                        {
                            *d += p * vv;
                        }
                    }
                }
            }
            let o_key = if harness
                .littlebit_tensors
                .contains_key(&format!("{lp}.self_attn.o_proj.weight"))
            {
                format!("{lp}.self_attn.o_proj.weight")
            } else {
                format!("{lp}.self_attn.out_proj.weight")
            };
            let ob = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &o_key,
                Some(&format!("{lp}.self_attn.o_proj.bias")),
                &attn_all,
                bsz,
                ops,
            )?;
            for (hi, &oi) in h_buf.iter_mut().zip(ob.iter()) {
                *hi += oi;
            }
        }

        let ffn_ln = first_vec_1d(
            &harness.vectors_1d,
            &[
                format!("{lp}.pre_feedforward_layernorm.weight"),
                format!("{lp}.post_attention_layernorm.weight"),
                format!("{lp}.ffn_norm.weight"),
                format!("{lp}.final_layer_norm.weight"),
            ],
        );
        let mut hffn = Vec::with_capacity(bsz * hidden_dim);
        for bi in 0..bsz {
            hffn.extend_from_slice(&rms_norm_cfg(
                &h_buf[bi * hidden_dim..(bi + 1) * hidden_dim],
                ffn_ln,
                rms_eps,
                gemma_norm_offset,
            ));
        }
        let db = if harness
            .littlebit_tensors
            .contains_key(&format!("{lp}.mlp.gate_proj.weight"))
        {
            let gb = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.mlp.gate_proj.weight"),
                None,
                &hffn,
                bsz,
                ops,
            )?;
            let ub = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.mlp.up_proj.weight"),
                None,
                &hffn,
                bsz,
                ops,
            )?;
            let gated: Vec<f32> = gb
                .iter()
                .zip(ub.iter())
                .map(|(&g, &u)| apply_gated_act(g, u, &hidden_act))
                .collect();
            run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.mlp.down_proj.weight"),
                None,
                &gated,
                bsz,
                ops,
            )?
        } else if harness
            .littlebit_tensors
            .contains_key(&format!("{lp}.feed_forward.w1.weight"))
        {
            let w1 = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.feed_forward.w1.weight"),
                None,
                &hffn,
                bsz,
                ops,
            )?;
            let w3 = run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.feed_forward.w3.weight"),
                None,
                &hffn,
                bsz,
                ops,
            )?;
            let gated: Vec<f32> = w1
                .iter()
                .zip(w3.iter())
                .map(|(&g, &u)| silu(g) * u)
                .collect();
            run_lb_batch(
                &harness.littlebit_tensors,
                &harness.vectors_1d,
                &format!("{lp}.feed_forward.w2.weight"),
                None,
                &gated,
                bsz,
                ops,
            )?
        } else {
            vec![0.0f32; bsz * hidden_dim]
        };
        for (hi, &di) in h_buf.iter_mut().zip(db.iter()) {
            *hi += di;
        }
        Ok(())
    };

    for r_idx in 0..bsz_total {
        let p_ids: Vec<usize> = tokenizer
            .encode(&prompts[r_idx], 512)
            .into_iter()
            .map(|id| id.min(vocab_size.saturating_sub(1)))
            .collect();
        req_prompt_lens.push(p_ids.len());

        let embed_t = harness
            .littlebit_tensors
            .get(&embed_key)
            .ok_or_else(|| "missing embed_tokens".to_string())?;
        let emb_ln = harness
            .vectors_1d
            .get(&format!("{pfx}embedding_norm.weight"))
            .map(Vec::as_slice);
        let plen = p_ids.len();
        let mut h_batch = Vec::with_capacity(plen * hidden_dim);
        for &tid in &p_ids {
            let mut row = reconstruct_embedding_row(embed_t, tid);
            if emb_ln.is_some() {
                row = rms_norm_cfg(&row, emb_ln, rms_eps, gemma_norm_offset);
            }
            h_batch.extend_from_slice(&row);
        }
        let prefill_map: Vec<(usize, usize)> = (0..plen).map(|bi| (r_idx, bi)).collect();
        for l in 0..num_layers {
            run_multi_req_layer(
                l,
                &mut h_batch,
                plen,
                &prefill_map,
                &mut req_k_caches,
                &mut req_v_caches,
                &mut req_conv_caches,
                &mut shared_ops,
            )?;
        }
        let final_ln = first_vec_1d(
            &harness.vectors_1d,
            &[
                format!("{pfx}norm.weight"),
                format!("{pfx}layer_norm.weight"),
            ],
        );
        let last_norm = rms_norm_cfg(
            &h_batch[(plen - 1) * hidden_dim..plen * hidden_dim],
            final_ln,
            rms_eps,
            gemma_norm_offset,
        );
        let mut l0 = run_lb_batch(
            &harness.littlebit_tensors,
            &harness.vectors_1d,
            &lm_head_key,
            None,
            &last_norm,
            1,
            &mut shared_ops,
        )?;
        let mut hist = p_ids;
        let first_tok = sample_next_token(&mut l0, &hist, params, &mut req_rng[r_idx]);
        hist.push(first_tok);
        req_gen_ids[r_idx].push(first_tok);
        req_cur_tok[r_idx] = first_tok;
        if max_new <= 1 {
            req_done[r_idx] = true;
        }
        req_histories.push(hist);
    }

    // 2. Lockstep Coalesced Multi-Request Decode Loop (`u_bits` read ONCE across all active requests!)
    for _step in 1..max_new {
        let active_reqs: Vec<usize> = (0..bsz_total).filter(|&i| !req_done[i]).collect();
        if active_reqs.is_empty() {
            break;
        }
        let b_act = active_reqs.len();
        let embed_t = harness
            .littlebit_tensors
            .get(&embed_key)
            .ok_or_else(|| "missing embed_tokens".to_string())?;
        let emb_ln = harness
            .vectors_1d
            .get(&format!("{pfx}embedding_norm.weight"))
            .map(Vec::as_slice);
        let mut h_act = Vec::with_capacity(b_act * hidden_dim);
        let mut step_map = Vec::with_capacity(b_act);
        for &ri in &active_reqs {
            let mut row = reconstruct_embedding_row(embed_t, req_cur_tok[ri]);
            if emb_ln.is_some() {
                row = rms_norm_cfg(&row, emb_ln, rms_eps, gemma_norm_offset);
            }
            h_act.extend_from_slice(&row);
            step_map.push((ri, req_histories[ri].len() - 1));
        }

        for l in 0..num_layers {
            run_multi_req_layer(
                l,
                &mut h_act,
                b_act,
                &step_map,
                &mut req_k_caches,
                &mut req_v_caches,
                &mut req_conv_caches,
                &mut shared_ops,
            )?;
        }

        let final_ln = first_vec_1d(
            &harness.vectors_1d,
            &[
                format!("{pfx}norm.weight"),
                format!("{pfx}layer_norm.weight"),
            ],
        );
        let mut h_final = Vec::with_capacity(b_act * hidden_dim);
        for ai in 0..b_act {
            h_final.extend_from_slice(&rms_norm_cfg(
                &h_act[ai * hidden_dim..(ai + 1) * hidden_dim],
                final_ln,
                rms_eps,
                gemma_norm_offset,
            ));
        }
        let mut logits_act = run_lb_batch(
            &harness.littlebit_tensors,
            &harness.vectors_1d,
            &lm_head_key,
            None,
            &h_final,
            b_act,
            &mut shared_ops,
        )?;

        for (ai, &ri) in active_reqs.iter().enumerate() {
            let l_sl = &mut logits_act[ai * vocab_size..(ai + 1) * vocab_size];
            let next_id = sample_next_token(l_sl, &req_histories[ri], params, &mut req_rng[ri]);
            if stop_ids.contains(&next_id) {
                req_done[ri] = true;
            } else {
                req_gen_ids[ri].push(next_id);
                req_histories[ri].push(next_id);
                req_cur_tok[ri] = next_id;
            }
        }
    }

    let elapsed_ns = (t0.elapsed().as_nanos() as u64).max(1);
    let mut total_gen = 0usize;
    let total_prompt: usize = req_prompt_lens.iter().sum();
    let mut items = Vec::with_capacity(bsz_total);
    for r_idx in 0..bsz_total {
        total_gen += req_gen_ids[r_idx].len();
        let text = tokenizer.decode(&req_gen_ids[r_idx], true);
        items.push(BatchedCausalItemReceipt {
            request_index: r_idx,
            prompt: prompts[r_idx].clone(),
            completion_tokens: req_gen_ids[r_idx].len(),
            prompt_tokens: req_prompt_lens[r_idx],
            generated_token_ids: std::mem::take(&mut req_gen_ids[r_idx]),
            generated_text: text,
        });
    }
    let agg_tps = (total_gen as f64) / ((elapsed_ns as f64) * 1e-9);

    Ok(BatchedCausalDecoderReceipt {
        model_id: model_id.to_string(),
        batch_size: bsz_total,
        num_layers,
        total_prompt_tokens: total_prompt,
        total_generated_tokens: total_gen,
        avg_bits_per_weight: avg_bpw,
        elapsed_ns,
        aggregate_tokens_per_sec: agg_tps,
        shared_batch_projections: shared_ops,
        items,
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
    fn test_bpe_tokenizer_gqa_rope_swiglu_and_mooncake_prefix_kv_reuse() {
        // 1. Verify GPT-2 byte-level BpeTokenizer encode/decode + special tokens
        let hf_tok_json = r#"{
            "model": {
                "vocab": {
                    "<|endoftext|>": 0,
                    "<|im_start|>": 1,
                    "<|im_end|>": 2,
                    "j": 3, "o": 4, "u": 5, "l": 6, "e": 7,
                    "Ġ": 8, "d": 9, "b": 10,
                    "jo": 11, "ul": 12, "joul": 13, "joule": 14,
                    "Ġd": 15, "Ġdb": 16
                },
                "merges": [
                    "j o",
                    "u l",
                    "jo ul",
                    "joul e",
                    "Ġ d",
                    "Ġd b"
                ]
            }
        }"#;
        let (vocab, merges) =
            BpeTokenizer::parse_hf_tokenizer_json(hf_tok_json).expect("parse bpe");
        let bpe = BpeTokenizer::from_vocab_and_merges(&vocab, &merges);
        let ids = bpe.encode("<|im_start|>joule db<|im_end|>", 64);
        assert_eq!(ids, vec![1, 14, 16, 2]);
        assert_eq!(bpe.decode(&ids, true), "joule db");
        assert_eq!(bpe.decode(&ids, false), "<|im_start|>joule db<|im_end|>");

        // 2. Build a synthetic 2-layer GQA (4 Q heads, 2 KV heads, head_dim=8, hidden=32, ffn=64) Llama/SmolLM model
        let d = 32usize;
        let kv_d = 16usize;
        let ffn = 64usize;
        let vsize = 32usize;
        let mk_2d = |name: &str, r: usize, c: usize, seed: f32| SafetensorBuildEntry {
            name: name.to_string(),
            dtype: SafetensorDtype::F32,
            shape: vec![r, c],
            values_f32: (0..(r * c))
                .map(|i| ((i as f32) * 0.19 + seed).cos() * 0.25)
                .collect(),
        };
        let mk_1d = |name: &str, n: usize, val: f32| SafetensorBuildEntry {
            name: name.to_string(),
            dtype: SafetensorDtype::F32,
            shape: vec![n],
            values_f32: vec![val; n],
        };

        let mut entries = vec![
            mk_2d("model.embed_tokens.weight", vsize, d, 0.1),
            mk_1d("model.norm.weight", d, 1.0),
        ];
        for l in 0..2 {
            let p = format!("model.layers.{l}");
            entries.extend([
                mk_1d(&format!("{p}.input_layernorm.weight"), d, 1.0),
                mk_2d(
                    &format!("{p}.self_attn.q_proj.weight"),
                    d,
                    d,
                    0.2 + l as f32,
                ),
                mk_2d(
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_d,
                    d,
                    0.3 + l as f32,
                ),
                mk_2d(
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_d,
                    d,
                    0.4 + l as f32,
                ),
                mk_2d(
                    &format!("{p}.self_attn.o_proj.weight"),
                    d,
                    d,
                    0.5 + l as f32,
                ),
                mk_1d(&format!("{p}.post_attention_layernorm.weight"), d, 1.0),
                mk_2d(&format!("{p}.mlp.gate_proj.weight"), ffn, d, 0.6 + l as f32),
                mk_2d(&format!("{p}.mlp.up_proj.weight"), ffn, d, 0.7 + l as f32),
                mk_2d(&format!("{p}.mlp.down_proj.weight"), d, ffn, 0.8 + l as f32),
            ]);
        }

        let raw = build_safetensors_bytes(&BTreeMap::new(), &entries).expect("safetensors");
        let mut harness = JouleHarnessEngine::new(16, 16, 16 * 1024 * 1024).expect("harness");
        ingest_safetensors_into_harness("smol-causal", &raw, 16, 4, &mut harness).expect("ingest");
        harness.model_vocabs.insert("smol-causal".into(), vocab);
        harness
            .model_bpe_merges
            .insert("smol-causal".into(), merges);
        harness.model_configs.insert(
            "smol-causal".into(),
            CausalModelMeta {
                num_attention_heads: Some(4),
                num_key_value_heads: Some(2),
                head_dim: Some(8),
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-5,
                bos_token_id: None,
                eos_token_ids: vec![],
                tie_word_embeddings: true,
                ..CausalModelMeta::default()
            },
        );

        let mut live_kv = LivePrefixKvStore::new(4);
        let params = CausalGenerationParams {
            max_new_tokens: 6,
            temperature: 0.0,
            ..CausalGenerationParams::default()
        };

        // Turn 1: cold prefill over 6-token prompt (`[1, 14, 16, 14, 16, 2]`) -> caches first 4-token block!
        let r1 = generate_with_littlebit_causal_decoder(
            &mut harness,
            &mut live_kv,
            "smol-causal",
            "<|im_start|>joule dbjoule db<|im_end|>",
            &params,
        )
        .expect("gen 1");
        assert_eq!(r1.num_layers, 2);
        assert_eq!(r1.num_q_heads, 4);
        assert_eq!(r1.num_kv_heads, 2);
        assert!(!r1.generated_token_ids.is_empty() && r1.generated_token_ids.len() <= 6);

        // Turn 2: identical prompt -> hits materialized 4-token prefix KV block (`4 * 14 = 56` fewer linear projections!)
        // AND produces bit-identical greedy tokens!
        let r2 = generate_with_littlebit_causal_decoder(
            &mut harness,
            &mut live_kv,
            "smol-causal",
            "<|im_start|>joule dbjoule db<|im_end|>",
            &params,
        )
        .expect("gen 2");
        assert_eq!(r1.generated_token_ids, r2.generated_token_ids);
        assert!(
            r2.linear_projections_executed + 50 <= r1.linear_projections_executed,
            "expected KV cache to skip >=56 prefill projections: r1={} vs r2={}",
            r1.linear_projections_executed,
            r2.linear_projections_executed
        );

        // Turn 3: Self-Speculative `LayerSkip` + Rank-8 Draft-and-Verify (`speculative_draft_k = 4`)
        // MUST be 100% lossless (bit-identical greedy token sequence to `r1`!)
        let spec_params = CausalGenerationParams {
            max_new_tokens: 6,
            temperature: 0.0,
            speculative_draft_k: 4,
            ..CausalGenerationParams::default()
        };
        let r3 = generate_with_littlebit_causal_decoder(
            &mut harness,
            &mut live_kv,
            "smol-causal",
            "<|im_start|>joule dbjoule db<|im_end|>",
            &spec_params,
        )
        .expect("gen speculative");
        assert_eq!(
            r3.generated_token_ids, r1.generated_token_ids,
            "Self-speculative decoding must produce bit-identical greedy tokens!"
        );
    }
}
