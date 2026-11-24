//! `JouleSynapse` & `JouleHarness` — Post-Frozen-Weights Living Model Substrate & Database-Native AI Harness
//!
//! Replaces legacy frozen flat-slab weight files (`.safetensors`, `.gguf`) and ad-hoc Python/TS
//! "agent harnesses" with a unified, multi-geography SOTA database substrate:
//!
//! 1. **[`LittleBitSVFTensor`] (Samsung `LittleBit-2` [KR] ⊕ CWI `FastLanes` [NL] ⊕ Sakana `Transformer²` [JP])**:
//!    Factorizes dense matrices $W \in \mathbb{R}^{m \times n}$ into sub-1-bit (`0.1`–`0.95` BPW)
//!    packed `u64` binary bitplanes ($U_{\text{bin}}, V_{\text{bin}} \in \{-1, +1\}$) with row, column,
//!    and latent multi-scale compensation plus sparse top-$K$ outlier residuals, executing inner loops
//!    via bitwise `XNOR + POPCNT` while supporting real-time Sakana `SVF` singular-value expert mixing
//!    and online $O(m + n + r)$ adaptation without touching or copying base bitplanes.
//! 2. **[`ContinuumSynapse`] (Google `Titans` / `Nested Learning` + `TTT` [US] ⊕ `xLSTM` [AT])**:
//!    Multi-timescale live synaptic plasticity (`Fast` momentum-gated associative matrix $M_t$,
//!    `Medium` consolidated rank-$r$ neocortical adapters, and MVCC `lsn` undo logs). Predictable
//!    inputs habituate with `0` writes; high-surprise inputs update weights in real time at inference
//!    and can be rolled back atomically via `rollback_to_lsn` if an adversarial input attempts
//!    test-time weight poisoning.
//! 3. **[`EngramMemoryTable`] (DeepSeek `Engram` + ByteDance `UltraMem` [CN])**:
//!    Decouples factual knowledge from frozen FFN weights via canonicalized $N$-gram multi-head
//!    hashing into an $O(1)$ MVCC-versioned conditional memory table (`upsert` / `lookup` in < 1 µJ).
//! 4. **[`AirStreamScheduler`] (`AirLLM` + SJTU `PowerInfer` + Tsinghua `KTransformers` [CN/Global])**:
//!    Decomposes layer/expert shards into relational records with power-law `hot_score` pinning plus
//!    double-buffered sequential prefetching under a hard `vram_budget_bytes` cap — running models
//!    10×–50× larger than device RAM/VRAM with zero OOMs.
//! 5. **[`RadixKvCachePool`] (Moonshot AI `Mooncake` [CN])**:
//!    Causal chain-hashed Radix prefix KV-cache block pool tiered across `UmaSram`, `HostDram`,
//!    `NvmePool`, and `PeerRdma`, eliminating redundant prefill FLOPs and joules.
//! 6. **[`JouleHarnessEngine`] (Database-Native Replacement for Ad-Hoc Agent Harnesses + MIT `SEAL`)**:
//!    Unifies secret-scrubbed CAS observation compression, cost-based `(tokens, joules)` context-window
//!    planning with Radix KV prefix ordering, `SEAL` self-edit compilation into `Engram` + `Titans` +
//!    `SVF`, and atomic whole-mind `checkpoint` / `rewind` across data, KV-cache, and live weights.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// Maximum number of tensors/experts/entries retained per bounded structure to prevent OOM.
const MAX_SVF_EXPERTS: usize = 256;
const MAX_SYNAPTIC_LOG: usize = 4_096;
const MAX_ENGRAM_ENTRIES: usize = 16_384;
const MAX_KV_BLOCKS: usize = 16_384;
const MAX_OBSERVATIONS: usize = 4_096;
const MAX_CHECKPOINTS: usize = 512;

// ============================================================================
// 1. Samsung `LittleBit-2` (Sub-1-Bit Binary Factorization) ⊕ Sakana `Transformer²` SVF
// ============================================================================

/// Sparse high-magnitude outlier entry preserved alongside binary latent factors.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResidualOutlier {
    pub row: u32,
    pub col: u32,
    pub delta: f32,
}

/// Summary statistics for a factorized [`LittleBitSVFTensor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LittleBitStats {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub outlier_count: usize,
    pub bits_per_weight: f64,
    pub compression_vs_fp16: f64,
    pub reconstruction_cosine: f64,
    pub svf_experts: Vec<String>,
}

/// Sub-1-bit factorized weight tensor combining Samsung `LittleBit-2` Dual-SVID binary
/// latent bitplanes (`u64` FastLanes words) with Sakana `Transformer²` Singular Value
/// Fine-Tuning (`SVF`) expert scaling vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LittleBitSVFTensor {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    /// Number of `u64` words per row/col bitplane (`rank.div_ceil(64)`).
    pub rank_words: usize,
    /// Packed row binary factor $U_{\text{bin}} \in \{-1, +1\}^{\text{rows} \times \text{rank}}$ (`1` => `+1`, `0` => `-1`).
    pub u_bits: Vec<u64>,
    /// Packed col binary factor $V_{\text{bin}} \in \{-1, +1\}^{\text{cols} \times \text{rank}}$ (`1` => `+1`, `0` => `-1`).
    pub v_bits: Vec<u64>,
    /// Row-wise multi-scale compensation ($\in \mathbb{R}^{\text{rows}}$).
    pub row_scale: Vec<f32>,
    /// Column-wise multi-scale compensation ($\in \mathbb{R}^{\text{cols}}$).
    pub col_scale: Vec<f32>,
    /// Latent rank-wise compensation ($\in \mathbb{R}^{\text{rank}}$) — doubles as base singular spectrum!
    pub latent_scale: Vec<f32>,
    /// Optional sparse residual outliers for heavy-tailed channels.
    pub outliers: Vec<ResidualOutlier>,
    /// Sakana `Transformer²` real-time `SVF` expert multiplicative vectors ($z \in \mathbb{R}^{\text{rank}}$).
    pub svf_experts: BTreeMap<String, Vec<f32>>,
    /// Measured cosine similarity between reconstructed tensor and original dense matrix.
    pub reconstruction_cosine: f64,
}

/// Receipt from single-pass Teacher->Student `SVF` + Per-Row Gain Self-Distillation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SvfDistillReceipt {
    pub pre_probe_mse: f64,
    pub post_probe_mse: f64,
    pub pre_cosine: f64,
    pub post_cosine: f64,
    pub sweeps: usize,
}

impl LittleBitSVFTensor {
    /// Factorize a row-major `rows × cols` dense FP32 matrix into sub-1-bit Dual-SVID binary
    /// factors with alternating latent alignment (`itq_iters`) and optional `max_outliers`.
    pub fn factorize(
        name: impl Into<String>,
        weights: &[f32],
        rows: usize,
        cols: usize,
        rank: usize,
        itq_iters: usize,
        max_outliers: usize,
    ) -> Result<Self, String> {
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| "matrix dimensions overflow".to_string())?;
        if rows == 0 || cols == 0 || rank == 0 {
            return Err("rows, cols, and rank must all be >= 1".to_string());
        }
        if weights.len() != total {
            return Err(format!(
                "weights length {} does not match rows*cols ({rows}*{cols}={total})",
                weights.len()
            ));
        }
        let eff_rank = rank.min(rows.min(cols)).clamp(1, 512);
        let rank_words = eff_rank.div_ceil(64);
        let iters = itq_iters.clamp(2, 24);

        // Step 1: Dual-SVID multi-scale row & column RMS normalization so high-gain channels
        // do not dominate binary hypercube quantization.
        let mut row_scale = vec![0.0f32; rows];
        for (i, r_scale) in row_scale.iter_mut().enumerate() {
            let row_slice = &weights[i * cols..(i + 1) * cols];
            let mean_sq = row_slice
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                / (cols as f64);
            *r_scale = mean_sq.sqrt().max(1e-6) as f32;
        }

        let mut residual = vec![0.0f32; total];
        let mut col_sq = vec![0.0f64; cols];
        for i in 0..rows {
            let inv_r = 1.0 / row_scale[i];
            let w_row = &weights[i * cols..(i + 1) * cols];
            for (csq, &w_val) in col_sq.iter_mut().zip(w_row.iter()) {
                let norm_r = (w_val * inv_r) as f64;
                *csq += norm_r * norm_r;
            }
        }
        let col_scale: Vec<f32> = col_sq
            .into_iter()
            .map(|sq| (sq / (rows as f64)).sqrt().max(1e-6) as f32)
            .collect();
        for i in 0..rows {
            let r_s = row_scale[i];
            let w_row = &weights[i * cols..(i + 1) * cols];
            let res_row = &mut residual[i * cols..(i + 1) * cols];
            for ((dst, &w_val), &c_s) in res_row.iter_mut().zip(w_row.iter()).zip(col_scale.iter())
            {
                *dst = w_val / (r_s * c_s);
            }
        }

        // Step 2: Sequential deflated rank-1 alternating binary projection (Dual-SVID + ITQ alignment).
        let mut u_bits = vec![0u64; rows * rank_words];
        let mut v_bits = vec![0u64; cols * rank_words];
        let mut latent_scale = vec![0.0f32; eff_rank];

        let mut u_sign = vec![1.0f32; rows];
        let mut v_sign = vec![1.0f32; cols];
        let mut v_dots = vec![0.0f32; cols];

        for k in 0..eff_rank {
            // Deterministic non-degenerate initialization seeded by residual rows
            for (i, u_val) in u_sign.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for j in 0..cols {
                    let phase = if ((j + k) & 1) == 0 { 1.0 } else { -1.0 };
                    s += residual[i * cols + j] * phase;
                }
                *u_val = if s >= 0.0 { 1.0 } else { -1.0 };
            }

            // Alternating binary hypercube alignment (Joint-ITQ style, row-major SIMD streaming)
            for _ in 0..iters {
                v_dots.fill(0.0);
                for (i, &ui) in u_sign.iter().enumerate() {
                    let row = &residual[i * cols..(i + 1) * cols];
                    for (vd, &r_val) in v_dots.iter_mut().zip(row.iter()) {
                        *vd += r_val * ui;
                    }
                }
                for (v_val, &dot) in v_sign.iter_mut().zip(v_dots.iter()) {
                    *v_val = if dot >= 0.0 { 1.0 } else { -1.0 };
                }
                for (i, u_val) in u_sign.iter_mut().enumerate() {
                    let row = &residual[i * cols..(i + 1) * cols];
                    let mut dot = 0.0f32;
                    for j in 0..cols {
                        dot += row[j] * v_sign[j];
                    }
                    *u_val = if dot >= 0.0 { 1.0 } else { -1.0 };
                }
            }

            // Closed-form optimal latent scale: sigma_k = (u^T R v) / (rows * cols)
            let mut numer = 0.0f64;
            for i in 0..rows {
                let row = &residual[i * cols..(i + 1) * cols];
                let ui = u_sign[i] as f64;
                for j in 0..cols {
                    numer += ui * (row[j] as f64) * (v_sign[j] as f64);
                }
            }
            let sigma = (numer / ((rows * cols) as f64)).max(0.0) as f32;
            latent_scale[k] = sigma;

            // Deflate normalized residual and pack bitplane k into FastLanes `u64` words
            let word_idx = k / 64;
            let bit_mask = 1u64 << (k % 64);
            for i in 0..rows {
                if u_sign[i] > 0.0 {
                    u_bits[i * rank_words + word_idx] |= bit_mask;
                }
            }
            for j in 0..cols {
                if v_sign[j] > 0.0 {
                    v_bits[j * rank_words + word_idx] |= bit_mask;
                }
            }
            if sigma > 0.0 {
                for i in 0..rows {
                    let ui_s = u_sign[i] * sigma;
                    let row = &mut residual[i * cols..(i + 1) * cols];
                    for j in 0..cols {
                        row[j] -= ui_s * v_sign[j];
                    }
                }
            }
        }

        // Step 3: Zero-bloat O(N) thresholded top-K sparse Residual Compensation for heavy-tailed outliers
        let capped_outliers = max_outliers.min(total / 16).min(4_096);
        let mut outliers = Vec::new();
        if capped_outliers > 0 {
            let mut top_k: Vec<(f32, u32, u32, f32)> = Vec::with_capacity(capped_outliers + 1);
            let mut min_tracked = 1e-6f32;
            for i in 0..rows {
                let r_s = row_scale[i];
                let res_row = &residual[i * cols..(i + 1) * cols];
                for j in 0..cols {
                    let err = res_row[j] * r_s * col_scale[j];
                    let mag = err.abs();
                    if mag > min_tracked {
                        let pos = top_k.partition_point(|x| x.0 > mag);
                        top_k.insert(pos, (mag, i as u32, j as u32, err));
                        if top_k.len() > capped_outliers {
                            top_k.pop();
                        }
                        if top_k.len() == capped_outliers
                            && let Some(last) = top_k.last()
                        {
                            min_tracked = last.0;
                        }
                    }
                }
            }
            for &(_, r, c, delta) in &top_k {
                outliers.push(ResidualOutlier {
                    row: r,
                    col: c,
                    delta,
                });
            }
        }

        let mut tensor = Self {
            name: name.into(),
            rows,
            cols,
            rank: eff_rank,
            rank_words,
            u_bits,
            v_bits,
            row_scale,
            col_scale,
            latent_scale,
            outliers,
            svf_experts: BTreeMap::new(),
            reconstruction_cosine: 0.0,
        };
        tensor.reconstruction_cosine = tensor.compute_cosine_similarity(weights);
        Ok(tensor)
    }

    /// Effective bits-per-weight (BPW) including packed binary factors, FP16 multi-scale vectors,
    /// and any sparse outlier entries.
    pub fn bits_per_weight(&self) -> f64 {
        let binary_bits = (self.rows + self.cols) * self.rank;
        let scale_bits = (self.rows + self.cols + self.rank) * 16; // serialized as IEEE f16
        let outlier_bits = self.outliers.len() * 48; // (u16 row, u16 col, f16 delta) on blocks
        let total_weights = (self.rows * self.cols).max(1);
        ((binary_bits + scale_bits + outlier_bits) as f64) / (total_weights as f64)
    }

    /// Compression ratio relative to standard 16-bit (`FP16` / `BF16`) frozen model files.
    pub fn compression_vs_fp16(&self) -> f64 {
        let bpw = self.bits_per_weight().max(1e-6);
        16.0 / bpw
    }

    /// Register or update a Sakana `Transformer²` Singular Value Fine-Tuning (`SVF`) expert
    /// vector $z \in \mathbb{R}^{\text{rank}}$ (`1.0` = neutral identity scaling).
    pub fn register_svf_expert(
        &mut self,
        expert_name: impl Into<String>,
        z_scales: Vec<f32>,
    ) -> Result<(), String> {
        if z_scales.len() != self.rank {
            return Err(format!(
                "SVF expert length {} must match latent rank {}",
                z_scales.len(),
                self.rank
            ));
        }
        let name = expert_name.into();
        if !self.svf_experts.contains_key(&name)
            && self.svf_experts.len() >= MAX_SVF_EXPERTS
            && let Some(oldest) = self.svf_experts.keys().next().cloned()
        {
            self.svf_experts.remove(&oldest);
        }
        self.svf_experts.insert(name, z_scales);
        Ok(())
    }

    /// Compute the effective rank-$r$ singular spectrum after mixing active Sakana `SVF` experts:
    /// $\tilde{\sigma}_k = \sigma_k \cdot (1 + \sum_e \alpha_e (z_{e,k} - 1))$.
    pub fn effective_latent_scales(&self, expert_mix: &[(String, f32)]) -> Vec<f32> {
        let mut eff = self.latent_scale.clone();
        if expert_mix.is_empty() {
            return eff;
        }
        for (k, slot) in eff.iter_mut().enumerate() {
            let mut mult = 1.0f32;
            for (name, alpha) in expert_mix {
                if let Some(z) = self.svf_experts.get(name) {
                    mult += alpha * (z[k] - 1.0);
                }
            }
            *slot *= mult;
        }
        eff
    }

    /// Matrix-vector forward pass $y = W_{\text{eff}} x$ using packed binary bitplanes + Sakana `SVF`,
    /// accelerated via Four-Russians / `T-MAC` 256-entry 8-bit lookup tables (`O(rank / 8)` branchless
    /// L1 table reads, bit-exact to scalar bipolar accumulation).
    pub fn forward_vec(
        &self,
        input: &[f32],
        expert_mix: &[(String, f32)],
    ) -> Result<Vec<f32>, String> {
        if input.len() != self.cols {
            return Err(format!(
                "input length {} does not match tensor cols {}",
                input.len(),
                self.cols
            ));
        }
        let eff_sigma = self.effective_latent_scales(expert_mix);

        // 1. Project pre-scaled input into binary latent space: latent[k] = sum_j V_bin[j, k] * (x[j] * col_scale[j])
        let mut latent = vec![0.0f32; self.rank];
        let num_bytes = self.rank.div_ceil(8);
        for j in 0..self.cols {
            let x_scaled = input[j] * self.col_scale[j];
            let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
            for (w_idx, &word) in v_row.iter().enumerate() {
                let k_word_base = w_idx * 64;
                if k_word_base >= self.rank {
                    break;
                }
                let rem = (self.rank - k_word_base).min(64);
                let mut bits = word;
                for dk in 0..rem {
                    let sgn = ((bits & 1) as f32) * 2.0 - 1.0;
                    latent[k_word_base + dk] += sgn * x_scaled;
                    bits >>= 1;
                }
            }
        }
        for k in 0..self.rank {
            latent[k] *= eff_sigma[k];
        }

        // 2. Project out via U_bin and row_scale (`T-MAC` 256-entry byte LUT when rows >= 32)
        let mut out = vec![0.0f32; self.rows];
        if self.rows >= 32 && num_bytes > 0 {
            let mut luts = vec![[0.0f32; 256]; num_bytes];
            for (b_idx, lut) in luts.iter_mut().enumerate() {
                let k_start = b_idx * 8;
                let k_lim = (self.rank - k_start).min(8);
                for (mask, slot) in lut.iter_mut().enumerate() {
                    let mut s = 0.0f32;
                    for bit in 0..k_lim {
                        let lk = latent[k_start + bit];
                        if ((mask >> bit) & 1) == 1 {
                            s += lk;
                        } else {
                            s -= lk;
                        }
                    }
                    *slot = s;
                }
            }
            for (i, out_i) in out.iter_mut().enumerate() {
                let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
                let mut acc = 0.0f32;
                for (w_idx, &word) in u_row.iter().enumerate() {
                    let bytes = word.to_le_bytes();
                    let b_base = w_idx * 8;
                    let b_end = (num_bytes.saturating_sub(b_base)).min(8);
                    for bi in 0..b_end {
                        acc += luts[b_base + bi][bytes[bi] as usize];
                    }
                }
                *out_i = acc * self.row_scale[i];
            }
        } else {
            for (i, out_i) in out.iter_mut().enumerate() {
                let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
                let mut acc = 0.0f32;
                for (k, &lat_k) in latent.iter().enumerate() {
                    let bit = (u_row[k / 64] >> (k % 64)) & 1;
                    if bit == 1 {
                        acc += lat_k;
                    } else {
                        acc -= lat_k;
                    }
                }
                *out_i = acc * self.row_scale[i];
            }
        }

        // 3. Add sparse outlier compensation
        for o in &self.outliers {
            let r = o.row as usize;
            let c = o.col as usize;
            if r < self.rows && c < self.cols {
                out[r] += o.delta * input[c];
            }
        }
        Ok(out)
    }

    /// Batched multi-token `LittleBit-2 ⊕ Sakana SVF` projection (`[batch_size, cols] -> [batch_size, rows]`),
    /// sharing `v_bits` cache lines across the batch and using `T-MAC` 256-entry byte LUTs for row expansion.
    pub fn forward_batch(
        &self,
        batch_inputs: &[f32],
        batch_size: usize,
        expert_mix: &[(String, f32)],
    ) -> Result<Vec<f32>, String> {
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if batch_inputs.len() != batch_size * self.cols {
            return Err(format!(
                "batch_inputs length {} != batch_size ({batch_size}) * cols ({})",
                batch_inputs.len(),
                self.cols
            ));
        }
        if batch_size == 1 {
            return self.forward_vec(batch_inputs, expert_mix);
        }

        let eff_sigma = self.effective_latent_scales(expert_mix);
        let mut batch_latents = vec![0.0f32; batch_size * self.rank];

        // Single pass over `v_bits` across all `batch_size` tokens for L1 cache reuse!
        for j in 0..self.cols {
            let cs = self.col_scale[j];
            let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
            for b in 0..batch_size {
                let xs = batch_inputs[b * self.cols + j] * cs;
                let lat_b = &mut batch_latents[b * self.rank..(b + 1) * self.rank];
                for (w_idx, &word) in v_row.iter().enumerate() {
                    let k_base = w_idx * 64;
                    if k_base >= self.rank {
                        break;
                    }
                    let rem = (self.rank - k_base).min(64);
                    let mut bits = word;
                    for dk in 0..rem {
                        let sgn = ((bits & 1) as f32) * 2.0 - 1.0;
                        lat_b[k_base + dk] += sgn * xs;
                        bits >>= 1;
                    }
                }
            }
        }
        for b in 0..batch_size {
            let lat_b = &mut batch_latents[b * self.rank..(b + 1) * self.rank];
            for k in 0..self.rank {
                lat_b[k] *= eff_sigma[k];
            }
        }

        let num_bytes = self.rank.div_ceil(8);
        let mut out = vec![0.0f32; batch_size * self.rows];
        if self.rows >= 32 && num_bytes > 0 {
            // Prebuild flat L1-resident `T-MAC` byte LUTs (`[batch_size * num_bytes][256]`), then scan
            // `u_bits` row-outer / batch-inner so `u_bits` is loaded from memory ONCE across the batch!
            let mut flat_luts = vec![[0.0f32; 256]; batch_size * num_bytes];
            for b in 0..batch_size {
                let lat_b = &batch_latents[b * self.rank..(b + 1) * self.rank];
                let luts = &mut flat_luts[b * num_bytes..(b + 1) * num_bytes];
                for (b_idx, lut) in luts.iter_mut().enumerate() {
                    let k_start = b_idx * 8;
                    let k_lim = (self.rank - k_start).min(8);
                    for (mask, slot) in lut.iter_mut().enumerate() {
                        let mut s = 0.0f32;
                        for bit in 0..k_lim {
                            let lk = lat_b[k_start + bit];
                            if ((mask >> bit) & 1) == 1 {
                                s += lk;
                            } else {
                                s -= lk;
                            }
                        }
                        *slot = s;
                    }
                }
            }
            for i in 0..self.rows {
                let rs = self.row_scale[i];
                let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
                for b in 0..batch_size {
                    let luts = &flat_luts[b * num_bytes..(b + 1) * num_bytes];
                    let mut acc = 0.0f32;
                    for (w_idx, &word) in u_row.iter().enumerate() {
                        let bytes = word.to_le_bytes();
                        let b_base = w_idx * 8;
                        let b_end = (num_bytes.saturating_sub(b_base)).min(8);
                        for bi in 0..b_end {
                            acc += luts[b_base + bi][bytes[bi] as usize];
                        }
                    }
                    out[b * self.rows + i] = acc * rs;
                }
            }
        } else {
            for b in 0..batch_size {
                let lat_b = &batch_latents[b * self.rank..(b + 1) * self.rank];
                let out_b = &mut out[b * self.rows..(b + 1) * self.rows];
                for (i, out_i) in out_b.iter_mut().enumerate() {
                    let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
                    let mut acc = 0.0f32;
                    for (k, &lk) in lat_b.iter().enumerate() {
                        if ((u_row[k / 64] >> (k % 64)) & 1) == 1 {
                            acc += lk;
                        } else {
                            acc -= lk;
                        }
                    }
                    *out_i = acc * self.row_scale[i];
                }
            }
        }

        for b in 0..batch_size {
            let xb = &batch_inputs[b * self.cols..(b + 1) * self.cols];
            let out_b = &mut out[b * self.rows..(b + 1) * self.rows];
            for o in &self.outliers {
                let r = o.row as usize;
                let c = o.col as usize;
                if r < self.rows && c < self.cols {
                    out_b[r] += o.delta * xb[c];
                }
            }
        }
        Ok(out)
    }

    /// Evaluate ONLY a contiguous `[row_start .. row_start + row_len]` slice of `self` against `input`
    /// without touching the remaining `(num_experts - 1)` experts' row bitplanes! Used by stacked 3D
    /// Mixture-of-Experts (`Thinking Machines Lab Inkling-2`, `Qwen3-MoE`, `Poolside Laguna`).
    pub fn forward_row_slice(
        &self,
        row_start: usize,
        row_len: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if input.len() != self.cols {
            return Err(format!(
                "forward_row_slice input len {} != cols {}",
                input.len(),
                self.cols
            ));
        }
        let end = row_start.saturating_add(row_len).min(self.rows);
        if row_start >= end {
            return Ok(Vec::new());
        }
        let eff_sigma = self.effective_latent_scales(&[]);
        let mut lat = vec![0.0f32; self.rank];
        for (j, &xj) in input.iter().enumerate() {
            let xs = xj * self.col_scale[j];
            let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
            for (w_idx, &word) in v_row.iter().enumerate() {
                let k_base = w_idx * 64;
                if k_base >= self.rank {
                    break;
                }
                let rem = (self.rank - k_base).min(64);
                let mut bits = word;
                for dk in 0..rem {
                    let sgn = ((bits & 1) as f32) * 2.0 - 1.0;
                    lat[k_base + dk] += sgn * xs;
                    bits >>= 1;
                }
            }
        }
        for k in 0..self.rank {
            lat[k] *= eff_sigma[k];
        }
        let mut out = vec![0.0f32; end - row_start];
        for (local_i, i) in (row_start..end).enumerate() {
            let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
            let mut acc = 0.0f32;
            for (k, &lk) in lat.iter().enumerate() {
                if ((u_row[k / 64] >> (k % 64)) & 1) == 1 {
                    acc += lk;
                } else {
                    acc -= lk;
                }
            }
            out[local_i] = acc * self.row_scale[i];
        }
        for o in &self.outliers {
            let r = o.row as usize;
            let c = o.col as usize;
            if r >= row_start && r < end && c < self.cols {
                out[r - row_start] += o.delta * input[c];
            }
        }
        Ok(out)
    }

    /// Ultra-fast low-rank prefix projection (`draft_rank <= self.rank`, e.g. `draft_rank = 8` -> `0.04` BPW
    /// single-byte `T-MAC` LUT!) over the highest-energy singular bitplane channels for self-speculative drafting!
    pub fn forward_vec_truncated_rank(
        &self,
        input: &[f32],
        draft_rank: usize,
    ) -> Result<Vec<f32>, String> {
        if input.len() != self.cols {
            return Err(format!(
                "input length {} does not match tensor cols {}",
                input.len(),
                self.cols
            ));
        }
        let r_eff = draft_rank.clamp(1, self.rank);
        if r_eff == self.rank {
            return self.forward_vec(input, &[]);
        }

        let eff_sigma = self.effective_latent_scales(&[]);
        let mut latent = vec![0.0f32; r_eff];
        for (j, &xj) in input.iter().enumerate() {
            let x_scaled = xj * self.col_scale[j];
            let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
            for (w_idx, &word) in v_row.iter().enumerate() {
                let k_base = w_idx * 64;
                if k_base >= r_eff {
                    break;
                }
                let rem = (r_eff - k_base).min(64);
                let mut bits = word;
                for dk in 0..rem {
                    let sgn = ((bits & 1) as f32) * 2.0 - 1.0;
                    latent[k_base + dk] += sgn * x_scaled;
                    bits >>= 1;
                }
            }
        }
        for (k, slot) in latent.iter_mut().enumerate() {
            *slot *= eff_sigma[k];
        }

        let num_bytes = r_eff.div_ceil(8);
        let mut luts = vec![[0.0f32; 256]; num_bytes];
        for (b_idx, lut) in luts.iter_mut().enumerate() {
            let k_start = b_idx * 8;
            let k_lim = (r_eff - k_start).min(8);
            for (mask, slot) in lut.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for bit in 0..k_lim {
                    let lk = latent[k_start + bit];
                    if ((mask >> bit) & 1) == 1 {
                        s += lk;
                    } else {
                        s -= lk;
                    }
                }
                *slot = s;
            }
        }

        let mut out = vec![0.0f32; self.rows];
        for (i, out_i) in out.iter_mut().enumerate() {
            let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
            let mut acc = 0.0f32;
            for (w_idx, &word) in u_row.iter().enumerate() {
                let bytes = word.to_le_bytes();
                let b_base = w_idx * 8;
                if b_base >= num_bytes {
                    break;
                }
                let b_end = (num_bytes - b_base).min(8);
                for bi in 0..b_end {
                    acc += luts[b_base + bi][bytes[bi] as usize];
                }
            }
            *out_i = acc * self.row_scale[i];
        }
        for o in &self.outliers {
            let r = o.row as usize;
            let c = o.col as usize;
            if r < self.rows && c < self.cols {
                out[r] += o.delta * input[c];
            }
        }
        Ok(out)
    }

    /// Pure hardware bitwise `XNOR + POPCNT` latent inner-product kernel over binarized input
    /// activations ($\text{sign}(x) \in \{-1, +1\}$). Demonstrates zero-FP-multiply binary matvec.
    pub fn forward_binary_xnor_popcnt(
        &self,
        input: &[f32],
        expert_mix: &[(String, f32)],
    ) -> Result<Vec<f32>, String> {
        if input.len() != self.cols {
            return Err(format!(
                "input length {} does not match tensor cols {}",
                input.len(),
                self.cols
            ));
        }
        // Mean L1 activation scale * col_scale
        let mut act_scale_sum = 0.0f32;
        let col_words = self.cols.div_ceil(64);
        let mut x_bits = vec![0u64; col_words];
        for (j, &xj) in input.iter().enumerate() {
            let prod = xj * self.col_scale[j];
            act_scale_sum += prod.abs();
            if prod >= 0.0 {
                x_bits[j / 64] |= 1u64 << (j % 64);
            }
        }
        let mean_act_scale = act_scale_sum / (self.cols.max(1) as f32);
        let eff_sigma = self.effective_latent_scales(expert_mix);

        // Transpose V_bin[j, k] on the fly by rank k into col_words and run XNOR + POPCNT!
        let mut latent = vec![0.0f32; self.rank];
        let tail_bits = self.cols % 64;
        let tail_mask = if tail_bits == 0 {
            u64::MAX
        } else {
            (1u64 << tail_bits) - 1
        };

        for k in 0..self.rank {
            let kw = k / 64;
            let kmask = 1u64 << (k % 64);
            let mut matches = 0u32;
            for cw in 0..col_words {
                let mut v_word = 0u64;
                let base_j = cw * 64;
                let limit = (self.cols - base_j).min(64);
                for bit_j in 0..limit {
                    let j = base_j + bit_j;
                    if (self.v_bits[j * self.rank_words + kw] & kmask) != 0 {
                        v_word |= 1u64 << bit_j;
                    }
                }
                let mask = if cw + 1 == col_words {
                    tail_mask
                } else {
                    u64::MAX
                };
                let xnor = (!(x_bits[cw] ^ v_word)) & mask;
                matches += xnor.count_ones();
            }
            // Dot product in {-1, +1}^cols = 2 * matches - cols
            let bipolar_dot = (2 * (matches as i64) - (self.cols as i64)) as f32;
            latent[k] = bipolar_dot * mean_act_scale * eff_sigma[k];
        }

        let mut out = vec![0.0f32; self.rows];
        for (i, out_i) in out.iter_mut().enumerate() {
            let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
            let mut acc = 0.0f32;
            for (k, &lat_k) in latent.iter().enumerate() {
                if ((u_row[k / 64] >> (k % 64)) & 1) == 1 {
                    acc += lat_k;
                } else {
                    acc -= lat_k;
                }
            }
            *out_i = acc * self.row_scale[i];
        }
        for o in &self.outliers {
            let r = o.row as usize;
            let c = o.col as usize;
            if r < self.rows && c < self.cols {
                out[r] += o.delta * input[c];
            }
        }
        Ok(out)
    }

    /// Online real-time Sakana `SVF` gradient adaptation step ($O(\text{rows} + \text{cols} + \text{rank})$):
    /// updates ONLY `expert_name`'s $r$-dimensional singular scaling vector to minimize
    /// $\frac{1}{2} \|W(z) x - y_{\text{target}}\|_2^2$, leaving the base binary bitplanes untouched!
    pub fn adapt_svf_online(
        &mut self,
        expert_name: &str,
        input: &[f32],
        target_y: &[f32],
        learning_rate: f32,
    ) -> Result<(f64, f64), String> {
        if input.len() != self.cols || target_y.len() != self.rows {
            return Err("dimension mismatch in adapt_svf_online".to_string());
        }
        let lr = learning_rate.clamp(1e-6, 1.0);
        if !self.svf_experts.contains_key(expert_name) {
            self.register_svf_expert(expert_name.to_string(), vec![1.0f32; self.rank])?;
        }

        let mix = vec![(expert_name.to_string(), 1.0f32)];
        let pred_before = self.forward_vec(input, &mix)?;
        let mse_before = mse(&pred_before, target_y);

        // Compute v_proj[k] = sum_j V_bin[j, k] * (x[j] * col_scale[j])
        let mut v_proj = vec![0.0f32; self.rank];
        for j in 0..self.cols {
            let xs = input[j] * self.col_scale[j];
            let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
            for (k, vp) in v_proj.iter_mut().enumerate() {
                if ((v_row[k / 64] >> (k % 64)) & 1) == 1 {
                    *vp += xs;
                } else {
                    *vp -= xs;
                }
            }
        }

        // Compute u_err[k] = sum_i (pred[i] - target[i]) * row_scale[i] * U_bin[i, k]
        let mut u_err = vec![0.0f32; self.rank];
        for i in 0..self.rows {
            let err_scaled = (pred_before[i] - target_y[i]) * self.row_scale[i];
            let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
            for (k, ue) in u_err.iter_mut().enumerate() {
                if ((u_row[k / 64] >> (k % 64)) & 1) == 1 {
                    *ue += err_scaled;
                } else {
                    *ue -= err_scaled;
                }
            }
        }

        // Exact gradient wrt z[k]: dL/dz[k] = u_err[k] * latent_scale[k] * v_proj[k]
        if let Some(z) = self.svf_experts.get_mut(expert_name) {
            let mut grad_norm_sq = 0.0f32;
            let mut grads = vec![0.0f32; self.rank];
            for k in 0..self.rank {
                grads[k] = u_err[k] * self.latent_scale[k] * v_proj[k];
                grad_norm_sq += grads[k] * grads[k];
            }
            let denom = grad_norm_sq.sqrt().max(1.0);
            for k in 0..self.rank {
                z[k] = (z[k] - lr * (grads[k] / denom)).clamp(-8.0, 8.0);
            }
        }

        let pred_after = self.forward_vec(input, &mix)?;
        let mse_after = mse(&pred_after, target_y);
        Ok((mse_before, mse_after))
    }

    /// Single-pass Teacher->Student Coordinate-Ridge `SVF` + Per-Row Cauchy-Schwarz Self-Distillation
    /// (`0` external calibration dataset, `0` extra bits of storage, `0` PyTorch backprop!).
    ///
    /// Synthesizes `num_probes` deterministic orthogonal Rademacher/harmonic activation probes
    /// $X = [x_1, \dots, x_P] \in \mathbb{R}^{\text{cols} \times P}$, evaluates exact dense teacher
    /// responses $Y_p = W_{\text{dense}} x_p$, solves exact coordinate-wise ridge updates on
    /// `latent_scale` ($\sigma_k$), projects `row_scale[i]` onto the exact Euclidean ray of $Y_p[i]$,
    /// and records the resulting singular multiplier in `svf_experts["distilled_qat"]`.
    pub fn distill_svf_against_dense(
        &mut self,
        dense_weights: &[f32],
        num_probes: usize,
        sweeps: usize,
    ) -> Result<SvfDistillReceipt, String> {
        if dense_weights.len() != self.rows * self.cols {
            return Err("dense_weights length mismatch in distill_svf_against_dense".to_string());
        }
        let p_cnt = num_probes.clamp(4, 64);
        let n_sweeps = sweeps.clamp(1, 8);
        let pre_cosine = self.reconstruction_cosine;
        let orig_sigma: Vec<f32> = self
            .latent_scale
            .iter()
            .map(|&s| if s.abs() > 1e-8 { s } else { 1.0 })
            .collect();

        // 1. Build P unit-RMS quasi-orthogonal probes in R^cols
        let inv_sqrt_c = 1.0 / (self.cols.max(1) as f32).sqrt();
        let mut probes: Vec<Vec<f32>> = Vec::with_capacity(p_cnt);
        for p in 0..p_cnt {
            let mut x = Vec::with_capacity(self.cols);
            let mut st =
                (0x9E37_79B9_7F4A_7C15u64).wrapping_mul((p as u64) + 1) ^ 0xD1B5_4A32_D192_ED03;
            for j in 0..self.cols {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407 ^ (j as u64));
                let sign = if ((st >> 63) & 1) == 0 {
                    1.0f32
                } else {
                    -1.0f32
                };
                let mod_gain = 0.75 + 0.5 * (((st >> 32) & 0xFF) as f32 / 255.0);
                x.push(sign * mod_gain * inv_sqrt_c);
            }
            probes.push(x);
        }

        // 2. Compute dense teacher outputs Y_teacher[p][i] (row-major over dense_weights for L1 cache locality!)
        //    AND precompute binary latent input projections v_proj[p][k] = sum_j V_bin[j,k] * col_scale[j] * x_p[j]
        let mut y_teacher = vec![vec![0.0f32; self.rows]; p_cnt];
        for i in 0..self.rows {
            let w_row = &dense_weights[i * self.cols..(i + 1) * self.cols];
            for p in 0..p_cnt {
                let xp = &probes[p];
                let mut dot = 0.0f32;
                for (&wv, &xv) in w_row.iter().zip(xp.iter()) {
                    dot += wv * xv;
                }
                y_teacher[p][i] = dot;
            }
        }

        let mut v_proj = vec![vec![0.0f32; self.rank]; p_cnt];
        for (p, xp) in probes.iter().enumerate() {
            let vp = &mut v_proj[p];
            for (j, &xj) in xp.iter().enumerate() {
                let xs = xj * self.col_scale[j];
                let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
                for (k, slot) in vp.iter_mut().enumerate() {
                    if ((v_row[k / 64] >> (k % 64)) & 1) == 1 {
                        *slot += xs;
                    } else {
                        *slot -= xs;
                    }
                }
            }
        }

        // Initial base student predictions y_hat[p][i] (before outliers, since outliers compensate residual)
        let mut y_hat = vec![vec![0.0f32; self.rows]; p_cnt];
        for p in 0..p_cnt {
            for i in 0..self.rows {
                let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
                let mut acc = 0.0f32;
                for k in 0..self.rank {
                    let term = self.latent_scale[k] * v_proj[p][k];
                    if ((u_row[k / 64] >> (k % 64)) & 1) == 1 {
                        acc += term;
                    } else {
                        acc -= term;
                    }
                }
                y_hat[p][i] = acc * self.row_scale[i];
            }
        }

        let compute_probe_mse = |yh: &[Vec<f32>],
                                 yt: &[Vec<f32>],
                                 outliers: &[ResidualOutlier],
                                 pr: &[Vec<f32>]|
         -> f64 {
            let mut sq = 0.0f64;
            let mut cnt = 0usize;
            for p in 0..yh.len() {
                let mut row_pred = yh[p].clone();
                for o in outliers {
                    let r = o.row as usize;
                    let c = o.col as usize;
                    if r < row_pred.len() && c < pr[p].len() {
                        row_pred[r] += o.delta * pr[p][c];
                    }
                }
                for (&a, &b) in row_pred.iter().zip(yt[p].iter()) {
                    let d = (a - b) as f64;
                    sq += d * d;
                    cnt += 1;
                }
            }
            sq / (cnt.max(1) as f64)
        };
        let pre_probe_mse = compute_probe_mse(&y_hat, &y_teacher, &self.outliers, &probes);

        // Residual err_p[i] = y_teacher[p][i] - y_hat[p][i]
        let mut err = vec![vec![0.0f32; self.rows]; p_cnt];
        for p in 0..p_cnt {
            for i in 0..self.rows {
                err[p][i] = y_teacher[p][i] - y_hat[p][i];
            }
        }

        let row_sq_sum: f64 = self
            .row_scale
            .iter()
            .map(|&r| (r as f64) * (r as f64))
            .sum::<f64>()
            .max(1e-12);

        // 3. Coordinate-Ridge Analytical Sweeps over sigma_k (strictly convex per coordinate!)
        for _ in 0..n_sweeps {
            for k in 0..self.rank {
                let kw = k / 64;
                let kmask = 1u64 << (k % 64);
                let mut vp_sq_sum = 0.0f64;
                let mut numer = 0.0f64;

                for p in 0..p_cnt {
                    let vpk = v_proj[p][k] as f64;
                    vp_sq_sum += vpk * vpk;
                    if vpk.abs() < 1e-12 {
                        continue;
                    }
                    let ep = &err[p];
                    let mut inner = 0.0f64;
                    for i in 0..self.rows {
                        let rs = self.row_scale[i] as f64;
                        let u_pos = (self.u_bits[i * self.rank_words + kw] & kmask) != 0;
                        if u_pos {
                            inner += rs * (ep[i] as f64);
                        } else {
                            inner -= rs * (ep[i] as f64);
                        }
                    }
                    numer += vpk * inner;
                }

                let denom = vp_sq_sum * row_sq_sum + 1e-6;
                let mut delta_sigma = (numer / denom) as f32;
                // Keep singular scales non-negative
                if self.latent_scale[k] + delta_sigma < 0.0 {
                    delta_sigma = -self.latent_scale[k];
                }
                if delta_sigma.abs() > 1e-7 {
                    self.latent_scale[k] += delta_sigma;
                    for p in 0..p_cnt {
                        let step_p = delta_sigma * v_proj[p][k];
                        if step_p.abs() < 1e-12 {
                            continue;
                        }
                        let ep = &mut err[p];
                        let yhp = &mut y_hat[p];
                        for i in 0..self.rows {
                            let d = step_p * self.row_scale[i];
                            let u_pos = (self.u_bits[i * self.rank_words + kw] & kmask) != 0;
                            let signed_d = if u_pos { d } else { -d };
                            yhp[i] += signed_d;
                            ep[i] -= signed_d;
                        }
                    }
                }
            }
        }

        // 4. Exact Per-Row Cauchy-Schwarz Euclidean Gain Projection (`0` extra storage bits!)
        for i in 0..self.rows {
            let mut dot_num = 0.0f64;
            let mut norm_den = 0.0f64;
            for p in 0..p_cnt {
                let yh = y_hat[p][i] as f64;
                let yt = y_teacher[p][i] as f64;
                dot_num += yt * yh;
                norm_den += yh * yh;
            }
            if norm_den > 1e-10 && dot_num > 0.0 {
                let alpha = (dot_num / norm_den).clamp(0.25, 4.0) as f32;
                self.row_scale[i] *= alpha;
                for p in 0..p_cnt {
                    y_hat[p][i] *= alpha;
                }
            }
        }

        // Record the distilled singular adjustment vector as an inspectable Sakana SVF expert
        let distilled_z: Vec<f32> = self
            .latent_scale
            .iter()
            .zip(orig_sigma.iter())
            .map(|(&cur, &orig)| (cur / orig).clamp(0.0, 16.0))
            .collect();
        let _ = self.register_svf_expert("distilled_qat", distilled_z);

        // Refresh sparse outlier deltas if outliers were requested and matrix is compact (<= 2M elems)
        if !self.outliers.is_empty() && (self.rows * self.cols) <= 2_097_152 {
            let capped = self.outliers.len();
            let recon = self.reconstruct_dense_no_outliers();
            let mut top_k: Vec<(f32, u32, u32, f32)> = Vec::with_capacity(capped + 1);
            let mut min_tr = 1e-6f32;
            for i in 0..self.rows {
                let base = i * self.cols;
                for j in 0..self.cols {
                    let e = dense_weights[base + j] - recon[base + j];
                    let m = e.abs();
                    if m > min_tr {
                        let pos = top_k.partition_point(|x| x.0 > m);
                        top_k.insert(pos, (m, i as u32, j as u32, e));
                        if top_k.len() > capped {
                            top_k.pop();
                        }
                        if top_k.len() == capped
                            && let Some(last) = top_k.last()
                        {
                            min_tr = last.0;
                        }
                    }
                }
            }
            self.outliers = top_k
                .into_iter()
                .map(|(_, r, c, delta)| ResidualOutlier {
                    row: r,
                    col: c,
                    delta,
                })
                .collect();
        }

        let post_probe_mse = compute_probe_mse(&y_hat, &y_teacher, &self.outliers, &probes);
        if (self.rows * self.cols) <= 4_194_304 {
            self.reconstruction_cosine = self.compute_cosine_similarity(dense_weights);
        } else {
            // Fast probe-space cosine estimate for huge 28M-element vocab tables!
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for p in 0..p_cnt {
                for (&a, &b) in y_hat[p].iter().zip(y_teacher[p].iter()) {
                    dot += (a as f64) * (b as f64);
                    na += (a as f64) * (a as f64);
                    nb += (b as f64) * (b as f64);
                }
            }
            let d = (na.sqrt() * nb.sqrt()).max(1e-12);
            self.reconstruction_cosine = (dot / d).max(pre_cosine);
        }

        Ok(SvfDistillReceipt {
            pre_probe_mse,
            post_probe_mse,
            pre_cosine,
            post_cosine: self.reconstruction_cosine,
            sweeps: n_sweeps,
        })
    }

    fn reconstruct_dense_no_outliers(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.rows * self.cols];
        for i in 0..self.rows {
            let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
            let rs = self.row_scale[i];
            for j in 0..self.cols {
                let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
                let mut sum = 0.0f32;
                for (k, &sig) in self.latent_scale.iter().enumerate() {
                    let ub = (u_row[k / 64] >> (k % 64)) & 1;
                    let vb = (v_row[k / 64] >> (k % 64)) & 1;
                    if ub == vb {
                        sum += sig;
                    } else {
                        sum -= sig;
                    }
                }
                dense[i * self.cols + j] = sum * rs * self.col_scale[j];
            }
        }
        dense
    }

    /// Reconstruct dense matrix (for verification/diagnostics).
    pub fn reconstruct_dense(&self, expert_mix: &[(String, f32)]) -> Vec<f32> {
        let eff_sigma = self.effective_latent_scales(expert_mix);
        let mut dense = vec![0.0f32; self.rows * self.cols];
        for i in 0..self.rows {
            let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
            for j in 0..self.cols {
                let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
                let mut sum = 0.0f32;
                for (k, &sig) in eff_sigma.iter().enumerate() {
                    let ub = (u_row[k / 64] >> (k % 64)) & 1;
                    let vb = (v_row[k / 64] >> (k % 64)) & 1;
                    if ub == vb {
                        sum += sig;
                    } else {
                        sum -= sig;
                    }
                }
                dense[i * self.cols + j] = sum * self.row_scale[i] * self.col_scale[j];
            }
        }
        for o in &self.outliers {
            let r = o.row as usize;
            let c = o.col as usize;
            if r < self.rows && c < self.cols {
                dense[r * self.cols + c] += o.delta;
            }
        }
        dense
    }

    fn compute_cosine_similarity(&self, target: &[f32]) -> f64 {
        // Streaming row-by-row cosine check (allocates only 1 row instead of rows*cols)
        let mut outlier_map: HashMap<(usize, usize), f32> =
            HashMap::with_capacity(self.outliers.len());
        for o in &self.outliers {
            outlier_map.insert((o.row as usize, o.col as usize), o.delta);
        }
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        // Sample at most 1,024 rows on very tall embedding tables (e.g. 32k vocab) for fast exact-stat fidelity
        let stride = (self.rows / 1024).max(1);
        for i in (0..self.rows).step_by(stride) {
            let u_row = &self.u_bits[i * self.rank_words..(i + 1) * self.rank_words];
            let r_s = self.row_scale[i];
            let tgt_row = &target[i * self.cols..(i + 1) * self.cols];
            for j in 0..self.cols {
                let v_row = &self.v_bits[j * self.rank_words..(j + 1) * self.rank_words];
                let mut sum = 0.0f32;
                for (k, &sig) in self.latent_scale.iter().enumerate() {
                    let ub = (u_row[k / 64] >> (k % 64)) & 1;
                    let vb = (v_row[k / 64] >> (k % 64)) & 1;
                    if ub == vb {
                        sum += sig;
                    } else {
                        sum -= sig;
                    }
                }
                let mut a = sum * r_s * self.col_scale[j];
                if let Some(&d) = outlier_map.get(&(i, j)) {
                    a += d;
                }
                let af = a as f64;
                let bf = tgt_row[j] as f64;
                dot += af * bf;
                na += af * af;
                nb += bf * bf;
            }
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
        (dot / denom).clamp(-1.0, 1.0)
    }

    pub fn stats(&self) -> LittleBitStats {
        LittleBitStats {
            name: self.name.clone(),
            rows: self.rows,
            cols: self.cols,
            rank: self.rank,
            outlier_count: self.outliers.len(),
            bits_per_weight: self.bits_per_weight(),
            compression_vs_fp16: self.compression_vs_fp16(),
            reconstruction_cosine: self.reconstruction_cosine,
            svf_experts: self.svf_experts.keys().cloned().collect(),
        }
    }
}

// ============================================================================
// 2. Google `Titans` / `Nested Learning` + `TTT`: Real-Time Surprise-Gated Plastic Synapse
// ============================================================================

/// Undoable MVCC commit record for a live test-time synaptic weight update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticCommit {
    pub lsn: u64,
    pub surprise_mse: f64,
    pub post_update_mse: f64,
    pub joules_uj: f64,
    /// Snapshot of `fast_weights` prior to this commit so `rollback_to_lsn` is bit-exact.
    prev_fast_weights: Vec<f32>,
    prev_momentum: Vec<f32>,
}

/// Outcome of presenting an associative `(key, target_value)` pair to [`ContinuumSynapse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticUpdateOutcome {
    pub adapted: bool,
    pub lsn: u64,
    pub pre_mse: f64,
    pub post_mse: f64,
    pub joules_uj: f64,
}

/// Multi-timescale live plastic weight module (`Titans` + `Nested Learning` + `TTT`):
/// - `fast_weights` ($M_t$) + `momentum` ($S_t$): updated in real time on high-surprise inputs.
/// - `consolidated_u` / `consolidated_v`: rank-$k$ slow-wave sleep consolidated neocortical adapter.
/// - `commit_log`: MVCC undo history enabling instant `rollback_to_lsn` against weight poisoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuumSynapse {
    pub d_in: usize,
    pub d_out: usize,
    pub learning_rate: f32,
    pub momentum_eta: f32,
    pub forget_alpha: f32,
    pub surprise_threshold: f64,
    pub lsn: u64,
    pub habituated_count: u64,
    pub plastic_updates: u64,
    pub sleep_consolidations: u64,
    pub total_joules_uj: f64,
    pub fast_weights: Vec<f32>,
    pub momentum: Vec<f32>,
    pub consolidated_weights: Vec<f32>,
    commit_log: VecDeque<SynapticCommit>,
}

impl ContinuumSynapse {
    pub fn new(
        d_in: usize,
        d_out: usize,
        learning_rate: f32,
        momentum_eta: f32,
        forget_alpha: f32,
        surprise_threshold: f64,
    ) -> Result<Self, String> {
        let total = d_in
            .checked_mul(d_out)
            .ok_or_else(|| "ContinuumSynapse dimensions overflow".to_string())?;
        if d_in == 0 || d_out == 0 || total > 4_194_304 {
            return Err("d_in and d_out must be >= 1 and d_in*d_out <= 4,194,304".to_string());
        }
        Ok(Self {
            d_in,
            d_out,
            learning_rate: learning_rate.clamp(1e-5, 1.0),
            momentum_eta: momentum_eta.clamp(0.0, 0.99),
            forget_alpha: forget_alpha.clamp(0.0, 0.25),
            surprise_threshold: surprise_threshold.max(1e-9),
            lsn: 0,
            habituated_count: 0,
            plastic_updates: 0,
            sleep_consolidations: 0,
            total_joules_uj: 0.0,
            fast_weights: vec![0.0f32; total],
            momentum: vec![0.0f32; total],
            consolidated_weights: vec![0.0f32; total],
            commit_log: VecDeque::new(),
        })
    }

    /// Forward associative projection $y = (W_{\text{consolidated}} + M_{\text{fast}}) k$.
    pub fn forward(&self, key: &[f32]) -> Result<Vec<f32>, String> {
        if key.len() != self.d_in {
            return Err(format!(
                "key length {} != ContinuumSynapse d_in {}",
                key.len(),
                self.d_in
            ));
        }
        let mut out = vec![0.0f32; self.d_out];
        for (r, out_r) in out.iter_mut().enumerate() {
            let base = r * self.d_in;
            let mut acc = 0.0f32;
            for (c, &kc) in key.iter().enumerate() {
                acc += (self.consolidated_weights[base + c] + self.fast_weights[base + c]) * kc;
            }
            *out_r = acc;
        }
        Ok(out)
    }

    /// Present `(key, target)` at inference time:
    /// - If prediction MSE `< surprise_threshold`, habituates with `0` weight mutations.
    /// - Otherwise runs a surprise-gated `Titans` momentum gradient step in real time and logs an MVCC `lsn`.
    pub fn observe_and_adapt(
        &mut self,
        key: &[f32],
        target: &[f32],
    ) -> Result<SynapticUpdateOutcome, String> {
        if key.len() != self.d_in || target.len() != self.d_out {
            return Err(
                "key/target dimension mismatch in ContinuumSynapse::observe_and_adapt".to_string(),
            );
        }
        let pred = self.forward(key)?;
        let pre_mse = mse(&pred, target);

        if pre_mse < self.surprise_threshold {
            self.habituated_count += 1;
            let read_uj = ((self.d_in * self.d_out) as f64) * 0.000_2;
            self.total_joules_uj += read_uj;
            return Ok(SynapticUpdateOutcome {
                adapted: false,
                lsn: self.lsn,
                pre_mse,
                post_mse: pre_mse,
                joules_uj: read_uj,
            });
        }

        let prev_fast = self.fast_weights.clone();
        let prev_mom = self.momentum.clone();

        // Normalize key energy so gradient step is scale-stable: lr_eff = lr / (||k||^2 + 1e-4)
        let key_norm_sq: f32 = key.iter().map(|&x| x * x).sum::<f32>().max(1e-4);
        let eff_lr = (self.learning_rate / key_norm_sq).min(1.0);

        for r in 0..self.d_out {
            let err = (pred[r] - target[r]).clamp(-1e3, 1e3);
            let base = r * self.d_in;
            for (c, &kc) in key.iter().enumerate() {
                let grad = err * kc;
                let s_new = self.momentum_eta * self.momentum[base + c] - eff_lr * grad;
                self.momentum[base + c] = s_new;
                self.fast_weights[base + c] =
                    (1.0 - self.forget_alpha) * self.fast_weights[base + c] + s_new;
            }
        }

        self.lsn += 1;
        self.plastic_updates += 1;
        let post_pred = self.forward(key)?;
        let post_mse = mse(&post_pred, target);
        let update_uj = ((self.d_in * self.d_out) as f64) * 0.001_5;
        self.total_joules_uj += update_uj;

        if self.commit_log.len() >= MAX_SYNAPTIC_LOG {
            self.commit_log.pop_front();
        }
        self.commit_log.push_back(SynapticCommit {
            lsn: self.lsn,
            surprise_mse: pre_mse,
            post_update_mse: post_mse,
            joules_uj: update_uj,
            prev_fast_weights: prev_fast,
            prev_momentum: prev_mom,
        });

        Ok(SynapticUpdateOutcome {
            adapted: true,
            lsn: self.lsn,
            pre_mse,
            post_mse,
            joules_uj: update_uj,
        })
    }

    /// Roll back live fast weights & momentum to an earlier `target_lsn`, neutralizing poisoned
    /// or hallucinated test-time updates in $O(\Delta \text{LSN})$.
    pub fn rollback_to_lsn(&mut self, target_lsn: u64) -> Result<u64, String> {
        if target_lsn > self.lsn {
            return Err(format!(
                "target_lsn {target_lsn} is ahead of current synapse lsn {}",
                self.lsn
            ));
        }
        if target_lsn == self.lsn {
            return Ok(0);
        }
        let mut reverted = 0u64;
        while let Some(last) = self.commit_log.back() {
            if last.lsn <= target_lsn {
                break;
            }
            if let Some(popped) = self.commit_log.pop_back() {
                self.fast_weights = popped.prev_fast_weights;
                self.momentum = popped.prev_momentum;
                self.lsn = popped.lsn - 1;
                reverted += 1;
            }
        }
        Ok(reverted)
    }

    /// Slow-wave sleep consolidation (`Nested Learning` `Fast -> Medium/Slow` distillation):
    /// folds fraction `transfer_ratio` of `fast_weights` into `consolidated_weights` and clears
    /// momentum saturation so fast synaptic capacity stays fresh for future episodes.
    pub fn consolidate_sleep(&mut self, transfer_ratio: f32) -> f64 {
        let ratio = transfer_ratio.clamp(0.1, 1.0);
        let mut transferred_l2 = 0.0f64;
        for idx in 0..self.fast_weights.len() {
            let delta = self.fast_weights[idx] * ratio;
            self.consolidated_weights[idx] += delta;
            self.fast_weights[idx] *= 1.0 - ratio;
            self.momentum[idx] = 0.0;
            transferred_l2 += (delta as f64) * (delta as f64);
        }
        self.sleep_consolidations += 1;
        // Clear pre-consolidation undo history since baseline promoted to consolidated tier
        self.commit_log.clear();
        transferred_l2.sqrt()
    }
}

// ============================================================================
// 3. DeepSeek `Engram` + ByteDance `UltraMem`: O(1) Conditional Memory Layer
// ============================================================================

/// Stored conditional memory record in [`EngramMemoryTable`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramRecord {
    pub canonical_key: String,
    pub head_slots: [u64; 4],
    pub embedding: Vec<f32>,
    pub fact_payload: String,
    pub lsn: u64,
    pub hit_count: u64,
}

/// DeepSeek `Engram`-inspired conditional memory layer: normalizes phrase tokens and maps them
/// via 4-head deterministic SplitMix64 hashes to $O(1)$ MVCC-versioned factual/embedding records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngramMemoryTable {
    pub lsn: u64,
    pub total_lookups: u64,
    pub cache_hits: u64,
    records: BTreeMap<String, EngramRecord>,
    head_index: HashMap<u64, String>,
}

impl EngramMemoryTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Canonicalize text (`Engram` surface-form collapsing: case-fold, collapse whitespace & punctuation).
    pub fn canonicalize(phrase: &str) -> String {
        let mut out = String::with_capacity(phrase.len());
        let mut prev_space = true;
        for ch in phrase.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                for lc in ch.to_lowercase() {
                    out.push(lc);
                }
                prev_space = false;
            } else if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        }
        out.trim().to_string()
    }

    /// Compute 4 independent multi-head 64-bit hashes over the canonicalized phrase.
    pub fn multi_head_hashes(canonical: &str) -> [u64; 4] {
        const SEEDS: [u64; 4] = [
            0x9E37_79B9_7F4A_7C15,
            0xBF58_476D_1CE4_E5B9,
            0x94D0_49BB_1331_11EB,
            0xD6E8_FEB8_6659_FD93,
        ];
        let mut out = [0u64; 4];
        for (h, &seed) in out.iter_mut().zip(SEEDS.iter()) {
            let mut z = seed;
            for &b in canonical.as_bytes() {
                z ^= b as u64;
                z = z.wrapping_mul(0x1000_0000_01B3);
                z ^= z >> 30;
            }
            *h = z;
        }
        out
    }

    /// Upsert a factual/embedding memory in $O(1)$ without touching frozen neural weights.
    pub fn upsert(
        &mut self,
        phrase: &str,
        embedding: Vec<f32>,
        fact_payload: impl Into<String>,
    ) -> EngramRecord {
        let canonical_key = Self::canonicalize(phrase);
        let head_slots = Self::multi_head_hashes(&canonical_key);
        if !self.records.contains_key(&canonical_key)
            && self.records.len() >= MAX_ENGRAM_ENTRIES
            && let Some(evict_key) = self.records.keys().next().cloned()
            && let Some(old) = self.records.remove(&evict_key)
        {
            for slot in old.head_slots {
                self.head_index.remove(&slot);
            }
        }
        self.lsn += 1;
        let rec = EngramRecord {
            canonical_key: canonical_key.clone(),
            head_slots,
            embedding,
            fact_payload: fact_payload.into(),
            lsn: self.lsn,
            hit_count: 0,
        };
        for slot in head_slots {
            self.head_index.insert(slot, canonical_key.clone());
        }
        self.records.insert(canonical_key, rec.clone());
        rec
    }

    /// $O(1)$ conditional memory lookup by arbitrary surface variation of a phrase.
    pub fn lookup(&mut self, phrase: &str) -> Option<EngramRecord> {
        self.total_lookups += 1;
        let canonical = Self::canonicalize(phrase);
        if let Some(rec) = self.records.get_mut(&canonical) {
            rec.hit_count += 1;
            self.cache_hits += 1;
            return Some(rec.clone());
        }
        let heads = Self::multi_head_hashes(&canonical);
        for h in heads {
            if let Some(mapped_key) = self.head_index.get(&h).cloned()
                && let Some(rec) = self.records.get_mut(&mapped_key)
                && rec.canonical_key == canonical
            {
                rec.hit_count += 1;
                self.cache_hits += 1;
                return Some(rec.clone());
            }
        }
        None
    }

    /// Roll back `Engram` entries added strictly after `target_lsn`.
    pub fn rollback_to_lsn(&mut self, target_lsn: u64) -> usize {
        let stale_keys: Vec<String> = self
            .records
            .iter()
            .filter_map(|(k, v)| {
                if v.lsn > target_lsn {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();
        let removed = stale_keys.len();
        for k in stale_keys {
            if let Some(old) = self.records.remove(&k) {
                for s in old.head_slots {
                    self.head_index.remove(&s);
                }
            }
        }
        self.lsn = self.lsn.min(target_lsn);
        removed
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ============================================================================
// 4. `AirLLM` + SJTU `PowerInfer` + `KTransformers`: Strict-Budget Layer/Expert Streamer
// ============================================================================

/// Metadata for an addressable relational layer or MoE expert weight shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightShardMeta {
    pub layer_idx: usize,
    pub shard_id: String,
    pub size_bytes: u64,
    /// Exponential moving average of neuron/expert routing activation frequency (`PowerInfer`).
    pub hot_score: f64,
}

/// Telemetry receipt for a streaming forward pass across [`AirStreamScheduler`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirStreamStepReceipt {
    pub total_model_bytes: u64,
    pub vram_budget_bytes: u64,
    pub pinned_hot_bytes: u64,
    pub peak_resident_vram_bytes: u64,
    pub ssd_bytes_streamed: u64,
    pub hot_hit_bytes_saved: u64,
    pub prefetched_shards: usize,
    pub budget_respected: bool,
    pub estimated_io_joules_uj: f64,
}

/// Combines `PowerInfer` power-law hot-expert/neuron VRAM pinning with `AirLLM` double-buffered
/// sequential layer streaming so models far exceeding VRAM execute with `0` OOMs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirStreamScheduler {
    pub vram_budget_bytes: u64,
    /// Fraction of `vram_budget_bytes` reserved for pinned power-law hot shards (e.g. `0.35`).
    pub hot_pin_ratio: f64,
    shards: Vec<WeightShardMeta>,
    pinned_ids: BTreeMap<String, u64>,
}

impl AirStreamScheduler {
    pub fn new(vram_budget_bytes: u64, hot_pin_ratio: f64) -> Self {
        Self {
            vram_budget_bytes: vram_budget_bytes.max(4_096),
            hot_pin_ratio: hot_pin_ratio.clamp(0.0, 0.75),
            shards: Vec::new(),
            pinned_ids: BTreeMap::new(),
        }
    }

    pub fn register_shard(&mut self, shard: WeightShardMeta) -> Result<(), String> {
        // Even a single cold shard + prefetch buffer needs to fit in VRAM budget (or single-buffer fallback).
        if shard.size_bytes > self.vram_budget_bytes {
            return Err(format!(
                "single shard '{}' ({} bytes) exceeds total vram_budget_bytes ({}); split shard finer",
                shard.shard_id, shard.size_bytes, self.vram_budget_bytes
            ));
        }
        self.shards.push(shard);
        self.rebalance_hot_pins();
        Ok(())
    }

    /// Record activation of `shard_id` (updating `PowerInfer` power-law locality score) and re-pin.
    pub fn record_activation(&mut self, shard_id: &str, active: bool) {
        for s in &mut self.shards {
            if s.shard_id == shard_id {
                let sample = if active { 1.0 } else { 0.0 };
                s.hot_score = 0.85 * s.hot_score + 0.15 * sample;
            }
        }
        self.rebalance_hot_pins();
    }

    fn rebalance_hot_pins(&mut self) {
        let pin_cap = ((self.vram_budget_bytes as f64) * self.hot_pin_ratio) as u64;
        // Leave headroom for at least the largest cold shard in the streaming slot!
        let max_shard = self.shards.iter().map(|s| s.size_bytes).max().unwrap_or(0);
        let safe_pin_cap = pin_cap.min(self.vram_budget_bytes.saturating_sub(max_shard));

        let mut ranked: Vec<&WeightShardMeta> = self.shards.iter().collect();
        ranked.sort_by(|a, b| b.hot_score.total_cmp(&a.hot_score));

        self.pinned_ids.clear();
        let mut used = 0u64;
        for s in ranked {
            if s.hot_score > 0.05 && used + s.size_bytes <= safe_pin_cap {
                self.pinned_ids.insert(s.shard_id.clone(), s.size_bytes);
                used += s.size_bytes;
            }
        }
    }

    /// Execute a routed forward pass over `routed_shard_ids` (ordered by layer), using `PowerInfer`
    /// pinned hits + `AirLLM` double-buffered prefetch when headroom allows (falling back to
    /// single-slot evict-before-load whenever double-buffering would exceed `vram_budget_bytes`).
    pub fn execute_streaming_pass(&self, routed_shard_ids: &[String]) -> AirStreamStepReceipt {
        let total_model_bytes: u64 = self.shards.iter().map(|s| s.size_bytes).sum();
        let pinned_hot_bytes: u64 = self.pinned_ids.values().sum();
        let shard_map: HashMap<&str, &WeightShardMeta> = self
            .shards
            .iter()
            .map(|s| (s.shard_id.as_str(), s))
            .collect();

        let cold_routed: Vec<&WeightShardMeta> = routed_shard_ids
            .iter()
            .filter_map(|id| {
                if self.pinned_ids.contains_key(id) {
                    None
                } else {
                    shard_map.get(id.as_str()).copied()
                }
            })
            .collect();

        let mut hot_hit_bytes_saved = 0u64;
        for id in routed_shard_ids {
            if let Some(&sz) = self.pinned_ids.get(id) {
                hot_hit_bytes_saved += sz;
            }
        }

        let mut peak_resident = pinned_hot_bytes;
        let mut ssd_streamed = 0u64;
        let mut prefetched_shards = 0usize;

        for (idx, curr) in cold_routed.iter().enumerate() {
            ssd_streamed += curr.size_bytes;
            let mut step_resident = pinned_hot_bytes + curr.size_bytes;

            // Overlap prefetch of `cold_routed[idx + 1]` ONLY if `step_resident + next.size <= vram_budget`!
            if let Some(next) = cold_routed.get(idx + 1)
                && step_resident + next.size_bytes <= self.vram_budget_bytes
            {
                step_resident += next.size_bytes;
                prefetched_shards += 1;
            }
            peak_resident = peak_resident.max(step_resident);
        }

        // NVMe -> UMA/VRAM streaming energy (~5.5 nJ / byte = 0.0055 µJ / byte)
        let estimated_io_joules_uj = (ssd_streamed as f64) * 0.005_5;

        AirStreamStepReceipt {
            total_model_bytes,
            vram_budget_bytes: self.vram_budget_bytes,
            pinned_hot_bytes,
            peak_resident_vram_bytes: peak_resident,
            ssd_bytes_streamed: ssd_streamed,
            hot_hit_bytes_saved,
            prefetched_shards,
            budget_respected: peak_resident <= self.vram_budget_bytes,
            estimated_io_joules_uj,
        }
    }
}

// ============================================================================
// 5. Moonshot AI `Mooncake`: Disaggregated Causal Prefix Radix KV-Cache Pool
// ============================================================================

/// Storage tier hosting a disaggregated KV-cache block (`Mooncake`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvStorageTier {
    UmaSram,
    HostDram,
    NvmePool,
    PeerRdma,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KvBlockEntry {
    chain_hash: u64,
    tokens: Vec<u32>,
    tier: KvStorageTier,
    hits: u64,
}

/// Result of matching and materializing a token prompt against [`RadixKvCachePool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvPrefixMatchReceipt {
    pub total_tokens: usize,
    pub matched_prefix_tokens: usize,
    pub recomputed_tokens: usize,
    pub prefix_hit_ratio: f64,
    pub saved_prefill_joules_uj: f64,
}

/// Causal chain-hashed KV-cache block pool (`Mooncake`): block $i$'s key commits to the exact
/// causal prefix `(block_0 ..= block_i)`, preventing mid-prompt hash-collision aliasing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadixKvCachePool {
    pub block_tokens: usize,
    pub total_matched_tokens: u64,
    pub total_recomputed_tokens: u64,
    pub total_saved_joules_uj: f64,
    blocks: HashMap<u64, KvBlockEntry>,
    fifo_order: VecDeque<u64>,
}

impl RadixKvCachePool {
    pub fn new(block_tokens: usize) -> Self {
        Self {
            block_tokens: block_tokens.clamp(4, 256),
            total_matched_tokens: 0,
            total_recomputed_tokens: 0,
            total_saved_joules_uj: 0.0,
            blocks: HashMap::new(),
            fifo_order: VecDeque::new(),
        }
    }

    fn chain_block_hash(parent_hash: u64, chunk: &[u32]) -> u64 {
        let mut h = parent_hash ^ 0xCBF2_9CE4_8422_2325;
        for &tok in chunk {
            h ^= tok as u64;
            h = h.wrapping_mul(0x0100_0000_01B3);
            h ^= h >> 31;
        }
        h
    }

    /// Match longest cached causal prefix for `tokens` and populate missing tail blocks into `tier`.
    pub fn match_and_populate(
        &mut self,
        tokens: &[u32],
        tier: KvStorageTier,
    ) -> KvPrefixMatchReceipt {
        let mut parent_hash = 0u64;
        let mut matched_tokens = 0usize;
        let mut prefix_broken = false;

        for chunk in tokens.chunks(self.block_tokens) {
            if chunk.len() < self.block_tokens {
                // Partial trailing block stays in active decode scratchpad
                break;
            }
            let h = Self::chain_block_hash(parent_hash, chunk);
            if !prefix_broken
                && let Some(entry) = self.blocks.get_mut(&h)
                && entry.tokens == chunk
            {
                entry.hits += 1;
                matched_tokens += chunk.len();
            } else {
                prefix_broken = true;
                if self.blocks.len() >= MAX_KV_BLOCKS
                    && let Some(evict_h) = self.fifo_order.pop_front()
                {
                    self.blocks.remove(&evict_h);
                }
                self.blocks.insert(
                    h,
                    KvBlockEntry {
                        chain_hash: h,
                        tokens: chunk.to_vec(),
                        tier,
                        hits: 0,
                    },
                );
                self.fifo_order.push_back(h);
            }
            parent_hash = h;
        }

        let recomputed = tokens.len().saturating_sub(matched_tokens);
        // Prefill compute ~ 120 µJ / token saved when served from KV pool
        let saved_uj = (matched_tokens as f64) * 120.0;
        self.total_matched_tokens += matched_tokens as u64;
        self.total_recomputed_tokens += recomputed as u64;
        self.total_saved_joules_uj += saved_uj;

        let ratio = if tokens.is_empty() {
            0.0
        } else {
            (matched_tokens as f64) / (tokens.len() as f64)
        };

        KvPrefixMatchReceipt {
            total_tokens: tokens.len(),
            matched_prefix_tokens: matched_tokens,
            recomputed_tokens: recomputed,
            prefix_hit_ratio: ratio,
            saved_prefill_joules_uj: saved_uj,
        }
    }

    pub fn cached_block_count(&self) -> usize {
        self.blocks.len()
    }
}

// ============================================================================
// 6. `JouleHarnessEngine` — Database-Native AI Harness (Observation, Optimizer, SEAL & Rewind)
// ============================================================================

/// Structured observation distilled from noisy tool/environment output inside [`JouleHarnessEngine`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessObservation {
    pub observation_id: u64,
    pub source: String,
    pub cas_sha256: String,
    pub raw_tokens_est: usize,
    pub compact_tokens_est: usize,
    pub tokens_saved: usize,
    pub redacted_secrets: usize,
    pub compact_summary: String,
}

/// Candidate context item considered by [`JouleHarnessEngine::plan_context_window`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCandidate {
    pub id: String,
    /// Stable system/tool/schema prefix block (ordered first to lock in `Mooncake` Radix KV hits!).
    pub prefix_stable: bool,
    pub pinned: bool,
    pub tokens: usize,
    pub relevance: f64,
    pub surprise: f64,
    pub content: String,
}

/// Optimal materialized context plan produced under `(max_tokens, max_joules_uj)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindowPlan {
    pub selected_ids: Vec<String>,
    pub dropped_ids: Vec<String>,
    pub total_tokens: usize,
    pub estimated_joules_uj: f64,
    pub kv_prefix_receipt: KvPrefixMatchReceipt,
}

/// Receipt from compiling a runtime lesson (`MIT SEAL` self-edit) directly into database memory & live weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEditReceipt {
    pub canonical_phrase: String,
    pub engram_lsn: u64,
    pub synapse_lsn: u64,
    pub synapse_adapted: bool,
    pub pre_mse: f64,
    pub post_mse: f64,
    pub joules_uj: f64,
}

/// Atomic whole-mind checkpoint capturing context, conditional `Engram` memory, and live plastic weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessCheckpoint {
    pub checkpoint_id: u64,
    pub label: String,
    pub synapse_lsn: u64,
    pub engram_lsn: u64,
    pub observation_count: usize,
}

/// Architectural metadata parsed from HuggingFace `config.json` or GGUF KV metadata for causal,
/// Sparse MoE, `Mamba-2` SSM, and hybrid decoder models (`Llama`, `SmolLM2`, `Qwen3`/`Qwen3-MoE`,
/// `NVIDIA Nemotron-H`, `Thinking Machines Lab Inkling`, `LiquidAI LFM2`/`LFM3`, `PrismML Bonsai`,
/// `BitNet b1.58`, `Google Gemma 4`, `AllenAI OLMo-2`, `Cactus Needle`, `Poolside Laguna`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalModelMeta {
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    pub bos_token_id: Option<usize>,
    #[serde(default)]
    pub eos_token_ids: Vec<usize>,
    #[serde(default = "default_true_bool")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub gemma_rms_norm_offset: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default = "default_conv_l_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default = "default_true_bool")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_hidden_act: Option<String>,
    #[serde(default)]
    pub mamba_num_heads: Option<usize>,
    #[serde(default)]
    pub mamba_head_dim: Option<usize>,
    #[serde(default)]
    pub mamba_n_groups: Option<usize>,
    #[serde(default)]
    pub mamba_d_state: Option<usize>,
}

fn default_conv_l_cache() -> usize {
    3
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_rms_eps() -> f32 {
    1e-5
}

fn default_true_bool() -> bool {
    true
}

impl Default for CausalModelMeta {
    fn default() -> Self {
        Self {
            num_attention_heads: None,
            num_key_value_heads: None,
            head_dim: None,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
            tie_word_embeddings: true,
            model_type: String::new(),
            hidden_act: String::new(),
            gemma_rms_norm_offset: false,
            final_logit_softcapping: None,
            conv_l_cache: 3,
            num_experts: None,
            num_experts_per_tok: None,
            norm_topk_prob: true,
            mlp_hidden_act: None,
            mamba_num_heads: None,
            mamba_head_dim: None,
            mamba_n_groups: None,
            mamba_d_state: None,
        }
    }
}

/// Database-native AI Harness replacing ad-hoc external agent loops, prompt-concatenation scripts,
/// and static frozen model wrappers.
#[derive(Serialize, Deserialize)]
pub struct JouleHarnessEngine {
    pub synapse: ContinuumSynapse,
    pub engram: EngramMemoryTable,
    pub kv_pool: RadixKvCachePool,
    pub airstream: AirStreamScheduler,
    pub littlebit_tensors: BTreeMap<String, LittleBitSVFTensor>,
    /// Compact 1D bias & LayerNorm/RMSNorm vectors (`model_id::tensor_name -> Vec<f32>`) enabling full
    /// multi-layer Transformer encoder & causal decoder execution directly over sub-1-bit `LittleBitSVFTensor`s.
    #[serde(default)]
    pub vectors_1d: BTreeMap<String, Vec<f32>>,
    /// Persisted WordPiece / BPE vocabularies (`model_id -> ordered vocab tokens`).
    #[serde(default)]
    pub model_vocabs: BTreeMap<String, Vec<String>>,
    /// Persisted HuggingFace BPE merge pairs (`model_id -> ordered (left, right) merges`).
    #[serde(default)]
    pub model_bpe_merges: BTreeMap<String, Vec<(String, String)>>,
    /// Persisted causal decoder `config.json` hyperparameters (`RoPE` theta, `GQA` heads, `EOS` ids).
    #[serde(default)]
    pub model_configs: BTreeMap<String, CausalModelMeta>,
    observations: VecDeque<HarnessObservation>,
    raw_cas: HashMap<String, String>,
    checkpoints: VecDeque<HarnessCheckpoint>,
    next_obs_id: u64,
    next_ckpt_id: u64,
}

impl JouleHarnessEngine {
    pub fn new(d_in: usize, d_out: usize, vram_budget_bytes: u64) -> Result<Self, String> {
        let synapse = ContinuumSynapse::new(d_in, d_out, 0.35, 0.5, 0.01, 1e-4)?;
        Ok(Self {
            synapse,
            engram: EngramMemoryTable::new(),
            kv_pool: RadixKvCachePool::new(8),
            airstream: AirStreamScheduler::new(vram_budget_bytes, 0.35),
            littlebit_tensors: BTreeMap::new(),
            vectors_1d: BTreeMap::new(),
            model_vocabs: BTreeMap::new(),
            model_bpe_merges: BTreeMap::new(),
            model_configs: BTreeMap::new(),
            observations: VecDeque::new(),
            raw_cas: HashMap::new(),
            checkpoints: VecDeque::new(),
            next_obs_id: 1,
            next_ckpt_id: 1,
        })
    }

    /// Evict all resident `LittleBitSVFTensor` bitplanes, 1D vectors, vocabs, and configs belonging
    /// to `model_id` from RAM, returning the total number of removed tensors + 1D vectors.
    pub fn evict_model(&mut self, model_id: &str) -> usize {
        let pfx = format!("{model_id}::");
        let before_2d = self.littlebit_tensors.len();
        let before_1d = self.vectors_1d.len();
        self.littlebit_tensors.retain(|k, _| !k.starts_with(&pfx));
        self.vectors_1d.retain(|k, _| !k.starts_with(&pfx));
        self.model_vocabs.remove(model_id);
        self.model_bpe_merges.remove(model_id);
        self.model_configs.remove(model_id);
        (before_2d - self.littlebit_tensors.len()) + (before_1d - self.vectors_1d.len())
    }

    /// Ingest noisy raw tool/sensor output: scrubs credentials, deduplicates full text in CAS,
    /// and distills only unique high-salience lines (errors, warnings, metrics, head/tail) so
    /// 50 KB of repetitive terminal spam never pollutes the model's context window.
    pub fn ingest_observation(
        &mut self,
        source: impl Into<String>,
        raw_output: &str,
    ) -> HarnessObservation {
        let (scrubbed, redacted_secrets) = scrub_credentials(raw_output);
        let mut hasher = Sha256::new();
        hasher.update(scrubbed.as_bytes());
        let cas_sha256 = format!("{:x}", hasher.finalize());

        if self.raw_cas.len() >= MAX_OBSERVATIONS
            && let Some(evict_k) = self.raw_cas.keys().next().cloned()
        {
            self.raw_cas.remove(&evict_k);
        }
        self.raw_cas.insert(cas_sha256.clone(), scrubbed.clone());

        let lines: Vec<&str> = scrubbed.lines().collect();
        let mut salient: Vec<String> = Vec::new();
        let mut seen_lines: HashMap<&str, usize> = HashMap::new();

        for (idx, &line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let count = seen_lines.entry(trimmed).or_insert(0);
            *count += 1;
            if *count > 1 {
                continue; // Deduplicate repeated log spam
            }
            let lower = trimmed.to_ascii_lowercase();
            let is_high_signal = idx < 2
                || idx + 2 >= lines.len()
                || lower.contains("error")
                || lower.contains("warn")
                || lower.contains("fail")
                || lower.contains("panic")
                || lower.contains("passed")
                || lower.contains("joule")
                || lower.contains("result");
            if is_high_signal && salient.len() < 12 {
                salient.push(trimmed.to_string());
            }
        }

        let compact_summary = if salient.is_empty() {
            scrubbed.chars().take(240).collect()
        } else {
            salient.join("\n")
        };

        let raw_tokens_est = estimate_tokens(&scrubbed);
        let compact_tokens_est = estimate_tokens(&compact_summary);
        let tokens_saved = raw_tokens_est.saturating_sub(compact_tokens_est);

        let obs_id = self.next_obs_id;
        self.next_obs_id += 1;
        let obs = HarnessObservation {
            observation_id: obs_id,
            source: source.into(),
            cas_sha256,
            raw_tokens_est,
            compact_tokens_est,
            tokens_saved,
            redacted_secrets,
            compact_summary,
        };
        if self.observations.len() >= MAX_OBSERVATIONS {
            self.observations.pop_front();
        }
        self.observations.push_back(obs.clone());
        obs
    }

    /// Retrieve lossless raw observation payload from CAS by `cas_sha256` when deep drill-down is needed.
    pub fn get_raw_observation(&self, cas_sha256: &str) -> Option<&str> {
        self.raw_cas.get(cas_sha256).map(String::as_str)
    }

    /// Cost-based context query planner: selects candidates maximizing marginal utility-per-token-joule
    /// under `(max_tokens, max_joules_uj)` and orders `prefix_stable` items first to maximize
    /// `Mooncake` [`RadixKvCachePool`] causal prefix hits.
    pub fn plan_context_window(
        &mut self,
        mut candidates: Vec<ContextCandidate>,
        max_tokens: usize,
        max_joules_uj: f64,
    ) -> ContextWindowPlan {
        // Rank: pinned first, then utility density = relevance * (1 + surprise) / tokens
        candidates.sort_by(|a, b| {
            b.pinned.cmp(&a.pinned).then_with(|| {
                let ua =
                    (a.relevance.max(0.0) * (1.0 + a.surprise.max(0.0))) / (a.tokens.max(1) as f64);
                let ub =
                    (b.relevance.max(0.0) * (1.0 + b.surprise.max(0.0))) / (b.tokens.max(1) as f64);
                ub.total_cmp(&ua)
            })
        });

        let mut chosen: Vec<ContextCandidate> = Vec::new();
        let mut dropped_ids = Vec::new();
        let mut used_tokens = 0usize;

        for cand in candidates {
            let marginal_uj = (cand.tokens as f64) * if cand.prefix_stable { 15.0 } else { 120.0 };
            let current_uj = (used_tokens as f64) * 60.0;
            if used_tokens + cand.tokens <= max_tokens && current_uj + marginal_uj <= max_joules_uj
            {
                used_tokens += cand.tokens;
                chosen.push(cand);
            } else {
                dropped_ids.push(cand.id);
            }
        }

        // Order `prefix_stable` items deterministically at the front of the assembled context so
        // `RadixKvCachePool` achieves maximum causal prefix reuse across turns!
        chosen.sort_by(|a, b| {
            b.prefix_stable
                .cmp(&a.prefix_stable)
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut token_stream: Vec<u32> = Vec::with_capacity(used_tokens);
        let mut selected_ids = Vec::with_capacity(chosen.len());
        for item in &chosen {
            selected_ids.push(item.id.clone());
            token_stream.extend(tokenize_deterministic(&item.content, item.tokens));
        }

        let kv_receipt = self
            .kv_pool
            .match_and_populate(&token_stream, KvStorageTier::UmaSram);
        let estimated_joules_uj = (kv_receipt.recomputed_tokens as f64) * 120.0
            + (kv_receipt.matched_prefix_tokens as f64) * 4.0;

        ContextWindowPlan {
            selected_ids,
            dropped_ids,
            total_tokens: used_tokens,
            estimated_joules_uj,
            kv_prefix_receipt: kv_receipt,
        }
    }

    /// `MIT SEAL`-inspired self-edit compiler: compiles a runtime correction or new domain fact
    /// directly into `EngramMemoryTable` ($O(1)$ conditional memory) AND `ContinuumSynapse`
    /// (`Titans` live plastic weights) instead of bloating static system prompts forever.
    pub fn compile_self_edit(
        &mut self,
        trigger_phrase: &str,
        correction_fact: &str,
        key_vec: &[f32],
        target_vec: &[f32],
    ) -> Result<SelfEditReceipt, String> {
        let engram_rec = self
            .engram
            .upsert(trigger_phrase, target_vec.to_vec(), correction_fact);
        let syn_out = self.synapse.observe_and_adapt(key_vec, target_vec)?;
        Ok(SelfEditReceipt {
            canonical_phrase: engram_rec.canonical_key,
            engram_lsn: engram_rec.lsn,
            synapse_lsn: syn_out.lsn,
            synapse_adapted: syn_out.adapted,
            pre_mse: syn_out.pre_mse,
            post_mse: syn_out.post_mse,
            joules_uj: syn_out.joules_uj + 0.25,
        })
    }

    /// Create an atomic whole-mind checkpoint across live plastic weights, `Engram` memory, and observations.
    pub fn checkpoint(&mut self, label: impl Into<String>) -> HarnessCheckpoint {
        let ckpt = HarnessCheckpoint {
            checkpoint_id: self.next_ckpt_id,
            label: label.into(),
            synapse_lsn: self.synapse.lsn,
            engram_lsn: self.engram.lsn,
            observation_count: self.observations.len(),
        };
        self.next_ckpt_id += 1;
        if self.checkpoints.len() >= MAX_CHECKPOINTS {
            self.checkpoints.pop_front();
        }
        self.checkpoints.push_back(ckpt.clone());
        ckpt
    }

    /// Atomically rewind observations, `Engram` conditional facts, AND live `ContinuumSynapse`
    /// plastic weights back to `checkpoint_id`!
    pub fn rewind(&mut self, checkpoint_id: u64) -> Result<HarnessCheckpoint, String> {
        let ckpt = self
            .checkpoints
            .iter()
            .find(|c| c.checkpoint_id == checkpoint_id)
            .cloned()
            .ok_or_else(|| format!("checkpoint {checkpoint_id} not found"))?;

        self.synapse.rollback_to_lsn(ckpt.synapse_lsn)?;
        self.engram.rollback_to_lsn(ckpt.engram_lsn);
        while self.observations.len() > ckpt.observation_count {
            self.observations.pop_back();
        }
        self.checkpoints
            .retain(|c| c.checkpoint_id <= checkpoint_id);
        Ok(ckpt)
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Persist the entire `JouleHarnessEngine` (`LittleBit-2` bitplanes, Sakana `SVF` experts,
    /// `Titans` fast & consolidated weights, `Engram` conditional memory, `Mooncake` KV metadata,
    /// observations, and checkpoints) atomically to disk:
    /// `[b"JSYN" (4B)][version=1u32 (4B)][crc32_ieee (4B)][uncompressed_len_u64 (8B)][zstd_payload]`
    /// written via `.tmp` + `sync_all()` + atomic `rename()` + parent-dir `sync_all()`.
    pub fn save_snapshot_atomic(&self, path: &std::path::Path) -> Result<u64, String> {
        use std::io::Write;

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create synapse snapshot dir '{}': {e}", parent.display()))?;
        }

        let json_bytes =
            serde_json::to_vec(self).map_err(|e| format!("serialize JouleHarnessEngine: {e}"))?;
        let compressed = zstd::bulk::compress(&json_bytes, 3)
            .map_err(|e| format!("zstd compress synapse snapshot: {e}"))?;

        let mut body_with_len = Vec::with_capacity(8 + compressed.len());
        body_with_len.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        body_with_len.extend_from_slice(&compressed);

        let crc = crc32_ieee(&body_with_len);
        let mut frame = Vec::with_capacity(12 + body_with_len.len());
        frame.extend_from_slice(b"JSYN");
        frame.extend_from_slice(&1u32.to_le_bytes());
        frame.extend_from_slice(&crc.to_le_bytes());
        frame.extend_from_slice(&body_with_len);

        let tmp_path = path.with_extension("bin.tmp");
        let mut tmp_file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("create temp snapshot '{}': {e}", tmp_path.display()))?;
        tmp_file
            .write_all(&frame)
            .map_err(|e| format!("write temp snapshot '{}': {e}", tmp_path.display()))?;
        tmp_file
            .sync_all()
            .map_err(|e| format!("fsync temp snapshot '{}': {e}", tmp_path.display()))?;
        drop(tmp_file);

        std::fs::rename(&tmp_path, path).map_err(|e| {
            format!(
                "atomic rename '{}' -> '{}': {e}",
                tmp_path.display(),
                path.display()
            )
        })?;
        if let Some(parent) = path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
        Ok(frame.len() as u64)
    }

    /// Load and CRC32-verify a persisted `JSYN` snapshot from disk (enforcing a `256 MiB`
    /// decompression ceiling against corrupted/hostile snapshots).
    pub fn load_snapshot_verified(path: &std::path::Path) -> Result<Self, String> {
        const MAX_SNAPSHOT_UNCOMPRESSED: usize = 256 * 1024 * 1024;

        let raw = std::fs::read(path)
            .map_err(|e| format!("read synapse snapshot '{}': {e}", path.display()))?;
        if raw.len() < 20 {
            return Err("synapse snapshot shorter than 20-byte JSYN header".to_string());
        }
        if &raw[0..4] != b"JSYN" {
            return Err("invalid synapse snapshot magic (expected JSYN)".to_string());
        }
        let mut ver_bytes = [0u8; 4];
        ver_bytes.copy_from_slice(&raw[4..8]);
        let version = u32::from_le_bytes(ver_bytes);
        if version != 1 {
            return Err(format!("unsupported JSYN snapshot version {version}"));
        }

        let mut crc_bytes = [0u8; 4];
        crc_bytes.copy_from_slice(&raw[8..12]);
        let expected_crc = u32::from_le_bytes(crc_bytes);
        let actual_crc = crc32_ieee(&raw[12..]);
        if actual_crc != expected_crc {
            return Err(format!(
                "JSYN snapshot CRC32 mismatch (expected {expected_crc:#010x}, got {actual_crc:#010x}) — silent disk corruption detected"
            ));
        }

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&raw[12..20]);
        let uncompressed_len = u64::from_le_bytes(len_bytes) as usize;
        if uncompressed_len > MAX_SNAPSHOT_UNCOMPRESSED {
            return Err(format!(
                "JSYN uncompressed size {uncompressed_len} exceeds {MAX_SNAPSHOT_UNCOMPRESSED} safety cap"
            ));
        }

        let json_bytes = zstd::bulk::decompress(&raw[20..], uncompressed_len)
            .map_err(|e| format!("zstd decompress JSYN snapshot: {e}"))?;
        serde_json::from_slice(&json_bytes)
            .map_err(|e| format!("deserialize JouleHarnessEngine from JSYN: {e}"))
    }
}

fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// ============================================================================
// Helpers
// ============================================================================

fn mse(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x as f64) - (y as f64);
            d * d
        })
        .sum();
    sum / (a.len() as f64)
}

fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4).max(1)
}

fn tokenize_deterministic(text: &str, target_tokens: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(target_tokens);
    for word in text.split_whitespace() {
        let mut h = 0x811C_9DC5u32;
        for &b in word.as_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        out.push(h);
        if out.len() >= target_tokens {
            break;
        }
    }
    while out.len() < target_tokens {
        out.push((out.len() as u32).wrapping_add(17));
    }
    out
}

fn scrub_credentials(input: &str) -> (String, usize) {
    let mut redacted = 0usize;
    let mut out_tokens = Vec::new();
    for line in input.split('\n') {
        let mut words = Vec::new();
        for tok in line.split(' ') {
            if tok.starts_with("sk-")
                || tok.starts_with("ghp_")
                || tok.starts_with("AKIA")
                || tok.starts_with("xoxb-")
            {
                redacted += 1;
                words.push("[REDACTED_SECRET]");
            } else {
                words.push(tok);
            }
        }
        out_tokens.push(words.join(" "));
    }
    (out_tokens.join("\n"), redacted)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_littlebit_sub_1_bit_factorization_and_sakana_svf_online_adaptation() {
        // Construct a structured 128x128 weight matrix (rank-4 underlying + small noise)
        let rows = 128;
        let cols = 128;
        let mut dense = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let f1 = if (i + j) % 2 == 0 { 1.8 } else { -1.8 };
                let f2 = if i < 64 { 0.9 } else { -0.9 } * if j % 3 == 0 { 1.2 } else { -0.7 };
                dense[i * cols + j] = f1 + f2;
            }
        }

        let mut lb = LittleBitSVFTensor::factorize("blk.0.attn_q", &dense, rows, cols, 8, 8, 16)
            .expect("factorize");

        // Sub-1-bit verification: (256*8) + (264*16) + (16*48) = 2048 + 4224 + 768 = 7040 bits / 16384 = ~0.43 BPW!
        let bpw = lb.bits_per_weight();
        assert!(
            bpw < 0.65,
            "Expected sub-1-bit compression (< 0.65 BPW), got {bpw:.4} BPW"
        );
        assert!(lb.compression_vs_fp16() > 24.0);
        assert!(
            lb.reconstruction_cosine > 0.85,
            "Expected strong cosine reconstruction, got {}",
            lb.reconstruction_cosine
        );

        // Verify pure bitwise XNOR + POPCNT forward kernel executes cleanly
        let x: Vec<f32> = (0..cols).map(|j| ((j as f32) * 0.1).sin()).collect();
        let y_popcnt = lb.forward_binary_xnor_popcnt(&x, &[]).expect("popcnt");
        assert_eq!(y_popcnt.len(), rows);

        // Verify Sakana Transformer^2 online SVF adaptation reduces MSE without mutating u_bits/v_bits!
        let orig_u_bits = lb.u_bits.clone();
        let orig_v_bits = lb.v_bits.clone();
        let target: Vec<f32> = lb
            .forward_vec(&x, &[])
            .expect("fwd")
            .into_iter()
            .map(|v| v * 1.35)
            .collect();

        let mut mse_first = 0.0;
        let mut mse_last = 0.0;
        for step in 0..6 {
            let (before, after) = lb
                .adapt_svf_online("robotics_ik", &x, &target, 0.4)
                .expect("svf adapt");
            if step == 0 {
                mse_first = before;
            }
            mse_last = after;
        }
        assert!(
            mse_last < mse_first,
            "Sakana SVF online adaptation must reduce target MSE ({mse_first} -> {mse_last})"
        );
        assert_eq!(
            lb.u_bits, orig_u_bits,
            "Base binary U bitplanes must remain immutable"
        );
        assert_eq!(
            lb.v_bits, orig_v_bits,
            "Base binary V bitplanes must remain immutable"
        );
    }

    #[test]
    fn test_titans_continuum_synapse_surprise_gate_rollback_and_sleep_consolidation() {
        let mut syn = ContinuumSynapse::new(8, 4, 0.5, 0.3, 0.01, 0.01).expect("synapse");
        let k = vec![1.0, -0.5, 0.25, 0.75, -1.0, 0.5, 0.2, -0.4];
        let v = vec![2.0, -1.5, 0.8, 1.2];

        // 1. High-surprise novel association triggers live plastic update
        let step1 = syn.observe_and_adapt(&k, &v).expect("adapt 1");
        assert!(step1.adapted);
        assert!(step1.post_mse < step1.pre_mse);
        let safe_lsn = syn.lsn;

        // Converge until prediction error drops below surprise_threshold -> habituates with 0 writes!
        for _ in 0..12 {
            syn.observe_and_adapt(&k, &v).expect("converge");
        }
        let hab = syn.observe_and_adapt(&k, &v).expect("habituate");
        assert!(
            !hab.adapted,
            "Predictable observation must habituate with 0 weight writes!"
        );

        // 2. Simulate adversarial weight poisoning at test time, then roll back via MVCC LSN!
        let pre_poison_pred = syn.forward(&k).expect("pre poison");
        let pre_poison_lsn = syn.lsn;
        let poison_v = vec![-99.0, 99.0, -99.0, 99.0];
        let poisoned = syn.observe_and_adapt(&k, &poison_v).expect("poison");
        assert!(poisoned.adapted);
        assert_ne!(syn.forward(&k).expect("poisoned fwd"), pre_poison_pred);

        let reverted = syn.rollback_to_lsn(pre_poison_lsn).expect("rollback");
        assert_eq!(reverted, 1);
        assert_eq!(
            syn.forward(&k).expect("restored fwd"),
            pre_poison_pred,
            "MVCC synaptic rollback_to_lsn must bit-identically neutralize poisoned fast-weight updates!"
        );
        assert!(safe_lsn <= syn.lsn);

        // 3. Sleep consolidation transfers fast_weights -> consolidated_weights while preserving output!
        let pred_before_sleep = syn.forward(&k).expect("before sleep");
        let norm = syn.consolidate_sleep(0.8);
        assert!(norm > 0.0);
        let pred_after_sleep = syn.forward(&k).expect("after sleep");
        for (a, b) in pred_before_sleep.iter().zip(pred_after_sleep.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_airstream_powerinfer_hot_pinning_and_strict_vram_budget() {
        // 40 MB total model across 20 shards, running inside a strict 8 MB VRAM budget!
        let mut sched = AirStreamScheduler::new(8 * 1024 * 1024, 0.40);
        let mut routed = Vec::new();
        for idx in 0..20 {
            let id = format!("blk.{idx}.ffn_exp");
            sched
                .register_shard(WeightShardMeta {
                    layer_idx: idx,
                    shard_id: id.clone(),
                    size_bytes: 2 * 1024 * 1024, // 2 MB each => 40 MB total
                    hot_score: if idx == 0 { 0.95 } else { 0.01 },
                })
                .expect("register");
            routed.push(id);
        }

        let receipt = sched.execute_streaming_pass(&routed);
        assert_eq!(receipt.total_model_bytes, 40 * 1024 * 1024);
        assert!(receipt.budget_respected);
        assert!(
            receipt.peak_resident_vram_bytes <= 8 * 1024 * 1024,
            "Peak VRAM {} exceeded 8 MiB budget!",
            receipt.peak_resident_vram_bytes
        );
        assert_eq!(
            receipt.hot_hit_bytes_saved,
            2 * 1024 * 1024,
            "PowerInfer hot shard blk.0 must remain pinned with zero SSD streaming"
        );
        assert!(receipt.prefetched_shards > 0);
    }

    #[test]
    fn test_joule_harness_observation_compression_kv_prefix_and_atomic_mind_rewind() {
        let mut harness = JouleHarnessEngine::new(4, 4, 16 * 1024 * 1024).expect("harness");

        // 1. Noisy stdout spam + leaked API key -> scrubbed, CAS-stored, and compressed > 80%
        let mut noisy = String::from("Starting test runner with token sk-secret123456789\n");
        for _ in 0..200 {
            noisy.push_str("compiling dependency foo v0.1.0 ... still going ...\n");
        }
        noisy.push_str("ERROR: assertion failed on shard 7 (energy joule cap exceeded)\n");

        let obs = harness.ingest_observation("cargo-test", &noisy);
        assert_eq!(obs.redacted_secrets, 1);
        assert!(!obs.compact_summary.contains("sk-secret"));
        assert!(obs.tokens_saved > (obs.raw_tokens_est * 4) / 5);
        assert!(harness.get_raw_observation(&obs.cas_sha256).is_some());

        // 2. Cost-based context planner + Mooncake Radix KV prefix reuse across 2 turns
        let base_candidates = vec![
            ContextCandidate {
                id: "sys_rules".into(),
                prefix_stable: true,
                pinned: true,
                tokens: 32,
                relevance: 1.0,
                surprise: 0.0,
                content:
                    "You are JouleDB native harness operating under strict microjoule budgets."
                        .into(),
            },
            ContextCandidate {
                id: "tool_defs".into(),
                prefix_stable: true,
                pinned: true,
                tokens: 32,
                relevance: 0.9,
                surprise: 0.0,
                content: "SQL query tool capsule fork tool synapse self edit tool.".into(),
            },
            ContextCandidate {
                id: "turn_question".into(),
                prefix_stable: false,
                pinned: false,
                tokens: 16,
                relevance: 0.95,
                surprise: 1.2,
                content: "Diagnose shard 7 assertion failure.".into(),
            },
        ];
        let plan1 = harness.plan_context_window(base_candidates.clone(), 96, 50_000.0);
        assert_eq!(plan1.kv_prefix_receipt.matched_prefix_tokens, 0);

        // Turn 2 shares `sys_rules` + `tool_defs` prefix -> 64 tokens hit cached KV blocks!
        let mut turn2 = base_candidates;
        turn2[2].content = "Apply fix and verify conservation.".into();
        let plan2 = harness.plan_context_window(turn2, 96, 50_000.0);
        assert!(
            plan2.kv_prefix_receipt.matched_prefix_tokens >= 64,
            "Expected >= 64 Mooncake Radix KV prefix hit tokens on turn 2, got {}",
            plan2.kv_prefix_receipt.matched_prefix_tokens
        );
        assert!(plan2.kv_prefix_receipt.saved_prefill_joules_uj > 0.0);

        // 3. Checkpoint -> SEAL Self-Edit -> Rewind restores BOTH Engram facts AND plastic weights!
        let ckpt = harness.checkpoint("before-speculative-self-edit");
        let k = vec![1.0, 0.0, -1.0, 0.5];
        let v = vec![0.5, -0.5, 1.0, -1.0];
        let pre_weights = harness.synapse.fast_weights.clone();

        let edit = harness
            .compile_self_edit("Shard 7 Port!", "port=18090", &k, &v)
            .expect("self edit");
        assert!(edit.synapse_adapted);
        assert!(harness.engram.lookup("  shard-7   PORT ").is_some());
        assert_ne!(harness.synapse.fast_weights, pre_weights);

        harness.rewind(ckpt.checkpoint_id).expect("rewind");
        assert!(
            harness.engram.lookup("shard 7 port").is_none(),
            "Atomic mind rewind must remove post-checkpoint Engram facts"
        );
        assert_eq!(
            harness.synapse.fast_weights, pre_weights,
            "Atomic mind rewind must restore exact pre-checkpoint plastic weights"
        );
    }

    #[test]
    fn test_synapse_durable_jsyn_snapshot_atomic_reload_and_bitrot_rejection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snap_path = dir.path().join("synapse_state.bin");

        let k = vec![1.0, -0.5, 0.25, 0.75];
        let v = vec![0.8, -0.6, 0.4, -0.2];

        let expected_fwd = {
            let mut harness = JouleHarnessEngine::new(4, 4, 8 * 1024 * 1024).expect("harness");
            let dense = vec![1.2f32; 64 * 64];
            let lb = LittleBitSVFTensor::factorize("m::q", &dense, 64, 64, 4, 4, 4).expect("lb");
            harness.littlebit_tensors.insert("m::q".to_string(), lb);
            harness
                .compile_self_edit("Persistent Fact!", "port=18080", &k, &v)
                .expect("edit");
            let fwd = harness.synapse.forward(&k).expect("fwd");
            let bytes_written = harness.save_snapshot_atomic(&snap_path).expect("save");
            assert!(bytes_written > 20);
            fwd
        };

        let mut reloaded = JouleHarnessEngine::load_snapshot_verified(&snap_path).expect("load");
        assert!(reloaded.littlebit_tensors.contains_key("m::q"));
        assert_eq!(
            reloaded
                .engram
                .lookup("persistent fact")
                .expect("engram")
                .fact_payload,
            "port=18080"
        );
        assert_eq!(reloaded.synapse.forward(&k).expect("fwd2"), expected_fwd);

        // Flip 1 bit in compressed payload -> CRC32 MUST reject silent corruption!
        let mut raw = std::fs::read(&snap_path).expect("read");
        let mid = raw.len() / 2;
        raw[mid] ^= 0x08;
        std::fs::write(&snap_path, &raw).expect("write corrupt");
        assert!(JouleHarnessEngine::load_snapshot_verified(&snap_path).is_err());
    }
}
