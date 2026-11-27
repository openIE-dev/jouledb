//! Orthogonal Harness LoRA Adapters & Rank-Concatenated Super-Adapters (`harness-hub` Parity)
//!
//! Implements the mathematical and runtime substrate inspired by Hugging Face `harness-hub`
//! (`Qwen3.8-27B-super-adapter`, *"Harness Adapters at Scale"*):
//! 1. **In-Harness Low-Rank Adapters (`HarnessLoRAAdapter`)**: Rank-$r$ (`B: rows × r`, `A: r × cols`,
//!    scaling $\gamma = \alpha / r$) adapters specialized for concrete agent harnesses (`pi`,
//!    `opencode`, `hermes`, `kiro-spec`, `jouledb-sql-ledger`).
//! 2. **Factorized $O((rows + cols) \cdot r_1 r_2)$ Subspace Orthogonality Audit
//!    (`audit_adapter_orthogonality`)**: Computes exact Frobenius cosine similarity
//!    $|\langle \Delta W_1, \Delta W_2 \rangle_F| / (\|\Delta W_1\|_F \|\Delta W_2\|_F)$ and row/column
//!    subspace overlaps via $\operatorname{Tr}((B_1^\top B_2)(A_2 A_1^\top))$ without materializing
//!    dense `rows × cols` matrices.
//! 3. **Gram-Schmidt Subspace De-Interference (`project_orthogonal_against`)**: Optional `O-LoRA`
//!    row-space orthogonal projection removing destructive cross-harness interference when two
//!    adapters exceed the cosine threshold.
//! 4. **Rank-Concatenated Super-Adapter (`compose_super_adapter`)**: Losslessly concatenates $K$
//!    specialist Rank-$r_i$ adapters along the rank dimension ($r_{\text{super}} = \sum_i r_i$,
//!    `combination_type = "cat"`, default weight $0.5$ per `Qwen3.8-27B-super-adapter`) and optionally
//!    fuses the resulting rank block directly onto a sub-1-bit `.jsyn` [`LittleBitSVFTensor`].
//! 5. **`AsyncGRPO` Rollout & Group-Advantage Exporter**: Computes group-normalized rewards
//!    $\hat{A}_i = (R_i - \mu_R) / (\sigma_R + \epsilon)$ over live database/MCP tool trajectories
//!    and updates adapter factors online or exports JSONL for `TRL` `AsyncGRPOTrainer`.

use crate::synapse::LittleBitSVFTensor;
use serde::{Deserialize, Serialize};

/// Combination strategy when merging multiple specialist harness adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuperAdapterCombineMode {
    /// Concatenate along the rank axis (`r_super = sum(r_i)`), preserving orthogonal subspaces
    /// without rank dilution (`Hugging Face PEFT` `combination_type="cat"`).
    RankConcat,
    /// Concatenate after Gram-Schmidt row-space orthogonalization whenever pairwise cosine exceeds
    /// `max_allowed_cosine`.
    OrthogonalizedRankConcat,
    /// Naive linear weighted sum into fixed rank (requires identical ranks; included as a baseline
    /// to quantify rank-dilution interference vs `RankConcat`).
    LinearAverageBaseline,
}

/// A Rank-$r$ LoRA adapter trained inside a specific agent harness (`ΔW = (lora_alpha / rank) * B * A`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessLoRAAdapter {
    /// Unique adapter identifier (e.g. `"opencode-r32"`, `"jouledb-sql-ledger-r32"`).
    pub adapter_id: String,
    /// Target agent harness family (`"pi"`, `"opencode"`, `"hermes"`, `"kiro"`, `"jouledb-mcp"`).
    pub harness_name: String,
    /// Target projection tensor name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub target_tensor: String,
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub lora_alpha: f32,
    /// Row-major `rank × cols` down-projection matrix $A$.
    pub lora_a: Vec<f32>,
    /// Row-major `rows × rank` up-projection matrix $B$.
    pub lora_b: Vec<f32>,
}

/// Pairwise subspace orthogonality report between two `HarnessLoRAAdapter`s on the same tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterOrthogonalityPair {
    pub adapter_lhs: String,
    pub adapter_rhs: String,
    /// Exact $|\langle \Delta W_1, \Delta W_2 \rangle_F| / (\|\Delta W_1\|_F \|\Delta W_2\|_F) \in [0, 1]$.
    pub frobenius_cosine: f64,
    /// Row-subspace overlap $\|A_1 A_2^\top\|_F / \sqrt{\|A_1 A_1^\top\|_F \|A_2 A_2^\top\|_F}$.
    pub input_subspace_cosine: f64,
    /// Column-subspace overlap $\|B_1^\top B_2\|_F / \sqrt{\|B_1^\top B_1\|_F \|B_2^\top B_2\|_F}$.
    pub output_subspace_cosine: f64,
    pub is_nearly_orthogonal: bool,
}

/// Summary audit across a set of $K$ specialist harness adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperAdapterOrthogonalityAudit {
    pub adapter_count: usize,
    pub total_concatenated_rank: usize,
    pub cosine_threshold: f64,
    pub max_pairwise_cosine: f64,
    pub mean_pairwise_cosine: f64,
    pub all_pairs_orthogonal: bool,
    pub pairs: Vec<AdapterOrthogonalityPair>,
}

/// Receipt emitted by [`compose_super_adapter`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperAdapterCompositionReceipt {
    pub super_adapter_id: String,
    pub mode: SuperAdapterCombineMode,
    pub constituent_adapters: Vec<String>,
    pub weights_applied: Vec<f32>,
    pub resulting_rank: usize,
    pub pre_audit: SuperAdapterOrthogonalityAudit,
    pub post_orthogonalization_max_cosine: f64,
    pub parameters_bytes_fp16: usize,
}

/// Single tool-call step recorded inside an agent harness trajectory for `AsyncGRPO` post-training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessRolloutStep {
    pub step_index: usize,
    pub tool_name: String,
    pub arguments_json: String,
    pub observation_summary: String,
    pub deterministic_ok: bool,
    pub energy_uj: u64,
    pub latency_us: u64,
}

/// Complete episode rollout for `AsyncGRPO` group-relative advantage calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessTrajectoryEpisode {
    pub episode_id: String,
    pub prompt_group_id: String,
    pub harness_name: String,
    pub prompt: String,
    pub steps: Vec<HarnessRolloutStep>,
    pub final_answer: String,
    /// Verifier reward (e.g. test pass + SQL invariant check - token/joule penalty).
    pub raw_reward: f64,
    /// Populated by [`compute_grpo_group_advantages`]: $(R_i - \mu_G) / (\sigma_G + 10^{-8})$.
    pub grpo_advantage: f64,
}

impl HarnessLoRAAdapter {
    /// Construct and validate a Rank-$r$ `HarnessLoRAAdapter`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adapter_id: impl Into<String>,
        harness_name: impl Into<String>,
        target_tensor: impl Into<String>,
        rows: usize,
        cols: usize,
        rank: usize,
        lora_alpha: f32,
        lora_a: Vec<f32>,
        lora_b: Vec<f32>,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rank == 0 {
            return Err("rows, cols, and rank must all be >= 1".to_string());
        }
        let expected_a = rank
            .checked_mul(cols)
            .ok_or_else(|| "rank * cols overflow".to_string())?;
        let expected_b = rows
            .checked_mul(rank)
            .ok_or_else(|| "rows * rank overflow".to_string())?;
        if lora_a.len() != expected_a {
            return Err(format!(
                "lora_a length {} does not match rank*cols ({expected_a})",
                lora_a.len()
            ));
        }
        if lora_b.len() != expected_b {
            return Err(format!(
                "lora_b length {} does not match rows*rank ({expected_b})",
                lora_b.len()
            ));
        }
        if !lora_alpha.is_finite() || lora_alpha == 0.0 {
            return Err("lora_alpha must be finite and non-zero".to_string());
        }
        Ok(Self {
            adapter_id: adapter_id.into(),
            harness_name: harness_name.into(),
            target_tensor: target_tensor.into(),
            rows,
            cols,
            rank,
            lora_alpha,
            lora_a,
            lora_b,
        })
    }

    /// Initialize a deterministic Kaiming-like $A$ + small directional $B$ seed for a named harness
    /// (useful for online `AsyncGRPO` updates and reproducible orthogonality tests).
    pub fn initialized_for_harness(
        adapter_id: impl Into<String>,
        harness_name: impl Into<String>,
        target_tensor: impl Into<String>,
        rows: usize,
        cols: usize,
        rank: usize,
        lora_alpha: f32,
        seed: u64,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rank == 0 {
            return Err("rows, cols, and rank must all be >= 1".to_string());
        }
        let expected_a = rank
            .checked_mul(cols)
            .ok_or_else(|| "rank * cols overflow".to_string())?;
        let expected_b = rows
            .checked_mul(rank)
            .ok_or_else(|| "rows * rank overflow".to_string())?;

        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_unit = || -> f32 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let frac = ((u >> 40) as f32) / ((1u32 << 24) as f32);
            frac * 2.0 - 1.0
        };

        let inv_sqrt_cols = 1.0 / (cols as f32).sqrt();
        let inv_sqrt_rows = 0.02 / (rows as f32).sqrt();
        let lora_a: Vec<f32> = (0..expected_a)
            .map(|_| next_unit() * inv_sqrt_cols)
            .collect();
        let lora_b: Vec<f32> = (0..expected_b)
            .map(|_| next_unit() * inv_sqrt_rows)
            .collect();

        Self::new(
            adapter_id,
            harness_name,
            target_tensor,
            rows,
            cols,
            rank,
            lora_alpha,
            lora_a,
            lora_b,
        )
    }

    /// Effective LoRA scaling $\gamma = \text{lora\_alpha} / \text{rank}$.
    #[inline]
    pub fn scaling(&self) -> f32 {
        self.lora_alpha / (self.rank as f32)
    }

    /// Evaluate adapter delta $\Delta y = \gamma B (A x) \in \mathbb{R}^{\text{rows}}$ in $O(r(\text{cols} + \text{rows}))$.
    pub fn forward_delta(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.cols {
            return Err(format!(
                "input length {} does not match adapter cols {}",
                input.len(),
                self.cols
            ));
        }
        let gamma = self.scaling();
        let mut latent = vec![0.0f32; self.rank];
        for (k, slot) in latent.iter_mut().enumerate() {
            let a_row = &self.lora_a[k * self.cols..(k + 1) * self.cols];
            let mut acc = 0.0f32;
            for (a_val, x_val) in a_row.iter().zip(input.iter()) {
                acc += a_val * x_val;
            }
            *slot = acc * gamma;
        }
        let mut out = vec![0.0f32; self.rows];
        for (i, out_slot) in out.iter_mut().enumerate() {
            let b_row = &self.lora_b[i * self.rank..(i + 1) * self.rank];
            let mut acc = 0.0f32;
            for (b_val, lat_val) in b_row.iter().zip(latent.iter()) {
                acc += b_val * lat_val;
            }
            *out_slot = acc;
        }
        Ok(out)
    }

    /// Perform an online advantage-weighted low-rank policy update (`AsyncGRPO`-style local step):
    /// nudges $\Delta W x$ toward `target_delta` when `grpo_advantage > 0` and away when `< 0`,
    /// clipped to `[-max_adv_clip, +max_adv_clip]` for stability.
    pub fn grpo_advantage_step(
        &mut self,
        input: &[f32],
        target_delta: &[f32],
        grpo_advantage: f32,
        learning_rate: f32,
    ) -> Result<f64, String> {
        if input.len() != self.cols {
            return Err("input dimension mismatch in grpo_advantage_step".to_string());
        }
        if target_delta.len() != self.rows {
            return Err("target_delta dimension mismatch in grpo_advantage_step".to_string());
        }
        let adv = grpo_advantage.clamp(-3.0, 3.0);
        let lr = learning_rate.clamp(1e-6, 0.25);
        let current = self.forward_delta(input)?;
        let gamma = self.scaling();

        // Residual error e = target_delta - current
        let mut residual = vec![0.0f32; self.rows];
        let mut mse = 0.0f64;
        for i in 0..self.rows {
            let diff = target_delta[i] - current[i];
            residual[i] = diff;
            mse += (diff as f64) * (diff as f64);
        }
        mse /= self.rows as f64;

        // Latent activations h = A x (length r)
        let mut h = vec![0.0f32; self.rank];
        for (k, h_slot) in h.iter_mut().enumerate() {
            let a_row = &self.lora_a[k * self.cols..(k + 1) * self.cols];
            let mut acc = 0.0f32;
            for (&av, &xv) in a_row.iter().zip(input.iter()) {
                acc += av * xv;
            }
            *h_slot = acc;
        }

        // Backprop residual weighted by GRPO advantage into B and A
        let step_scale = lr * adv * gamma;
        let mut grad_h = vec![0.0f32; self.rank];
        for (i, &res_i) in residual.iter().enumerate() {
            let b_row = &mut self.lora_b[i * self.rank..(i + 1) * self.rank];
            for k in 0..self.rank {
                grad_h[k] += b_row[k] * res_i;
                b_row[k] += step_scale * res_i * h[k];
            }
        }
        for (k, &gh_k) in grad_h.iter().enumerate() {
            let a_row = &mut self.lora_a[k * self.cols..(k + 1) * self.cols];
            let coeff = step_scale * gh_k;
            for (av, &xv) in a_row.iter_mut().zip(input.iter()) {
                *av += coeff * xv;
            }
        }
        Ok(mse)
    }

    /// Project this adapter's row-space ($A$) onto the orthogonal complement of `other.lora_a`
    /// (`O-LoRA` Gram-Schmidt projection), eliminating cross-harness subspace interference.
    pub fn project_orthogonal_against(&mut self, other: &HarnessLoRAAdapter) -> Result<(), String> {
        if self.cols != other.cols {
            return Err("adapter cols must match for orthogonal projection".to_string());
        }
        // Build an orthonormal basis Q for rowspan(other.lora_a)
        let mut basis: Vec<Vec<f32>> = Vec::with_capacity(other.rank);
        for k in 0..other.rank {
            let mut v = other.lora_a[k * other.cols..(k + 1) * other.cols].to_vec();
            for q in &basis {
                let dot: f64 = v
                    .iter()
                    .zip(q.iter())
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum();
                for (vi, &qi) in v.iter_mut().zip(q.iter()) {
                    *vi -= (dot as f32) * qi;
                }
            }
            let norm_sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
            if norm_sq > 1e-12 {
                let inv = (1.0 / norm_sq.sqrt()) as f32;
                for vi in &mut v {
                    *vi *= inv;
                }
                basis.push(v);
            }
        }

        // Remove projection onto Q from each row of self.lora_a (two-pass classical Gram-Schmidt
        // + explicit nullspace zeroing when a row lies completely inside rowspan(other.lora_a)).
        for k in 0..self.rank {
            let row = &mut self.lora_a[k * self.cols..(k + 1) * self.cols];
            let orig_norm_sq: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
            for _pass in 0..2 {
                for q in &basis {
                    let dot: f64 = row
                        .iter()
                        .zip(q.iter())
                        .map(|(&a, &b)| (a as f64) * (b as f64))
                        .sum();
                    for (ri, &qi) in row.iter_mut().zip(q.iter()) {
                        *ri -= (dot as f32) * qi;
                    }
                }
            }
            let rem_norm_sq: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
            if rem_norm_sq <= orig_norm_sq * 1e-10 {
                row.fill(0.0);
            }
        }
        Ok(())
    }
}

/// Compute `M_A = A_1 * A_2^T` (`r1 × r2`) in $O(r_1 r_2 \cdot \text{cols})$.
fn gram_rows(a1: &[f32], r1: usize, a2: &[f32], r2: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r1 * r2];
    for i in 0..r1 {
        let row1 = &a1[i * cols..(i + 1) * cols];
        for j in 0..r2 {
            let row2 = &a2[j * cols..(j + 1) * cols];
            let mut dot = 0.0f64;
            for (&x, &y) in row1.iter().zip(row2.iter()) {
                dot += (x as f64) * (y as f64);
            }
            out[i * r2 + j] = dot;
        }
    }
    out
}

/// Compute `M_B = B_1^T * B_2` (`r1 × r2`) in $O(r_1 r_2 \cdot \text{rows})$.
fn gram_cols(b1: &[f32], r1: usize, b2: &[f32], r2: usize, rows: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r1 * r2];
    for row in 0..rows {
        let s1 = &b1[row * r1..(row + 1) * r1];
        let s2 = &b2[row * r2..(row + 1) * r2];
        for (i, &v1) in s1.iter().enumerate() {
            let v1d = v1 as f64;
            let out_row = &mut out[i * r2..(i + 1) * r2];
            for (j, &v2) in s2.iter().enumerate() {
                out_row[j] += v1d * (v2 as f64);
            }
        }
    }
    out
}

/// Exact factorized Frobenius inner product $\langle \Delta W_1, \Delta W_2 \rangle_F$ and
/// row/column subspace cosines without allocating `rows × cols`.
pub fn pairwise_adapter_orthogonality(
    lhs: &HarnessLoRAAdapter,
    rhs: &HarnessLoRAAdapter,
    cosine_threshold: f64,
) -> Result<AdapterOrthogonalityPair, String> {
    if lhs.rows != rhs.rows || lhs.cols != rhs.cols {
        return Err(format!(
            "dimension mismatch: {}x{} vs {}x{}",
            lhs.rows, lhs.cols, rhs.rows, rhs.cols
        ));
    }

    let ma_12 = gram_rows(&lhs.lora_a, lhs.rank, &rhs.lora_a, rhs.rank, lhs.cols);
    let mb_12 = gram_cols(&lhs.lora_b, lhs.rank, &rhs.lora_b, rhs.rank, lhs.rows);
    let ma_11 = gram_rows(&lhs.lora_a, lhs.rank, &lhs.lora_a, lhs.rank, lhs.cols);
    let mb_11 = gram_cols(&lhs.lora_b, lhs.rank, &lhs.lora_b, lhs.rank, lhs.rows);
    let ma_22 = gram_rows(&rhs.lora_a, rhs.rank, &rhs.lora_a, rhs.rank, rhs.cols);
    let mb_22 = gram_cols(&rhs.lora_b, rhs.rank, &rhs.lora_b, rhs.rank, rhs.rows);

    let g1 = lhs.scaling() as f64;
    let g2 = rhs.scaling() as f64;

    let inner_12: f64 = g1
        * g2
        * ma_12
            .iter()
            .zip(mb_12.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let norm1_sq: f64 = g1
        * g1
        * ma_11
            .iter()
            .zip(mb_11.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .max(0.0);
    let norm2_sq: f64 = g2
        * g2
        * ma_22
            .iter()
            .zip(mb_22.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .max(0.0);

    let denom = (norm1_sq.sqrt() * norm2_sq.sqrt()).max(1e-18);
    let frob_cos = (inner_12.abs() / denom).clamp(0.0, 1.0);

    let a12_frob = ma_12.iter().map(|x| x * x).sum::<f64>().sqrt();
    let a11_frob = ma_11.iter().map(|x| x * x).sum::<f64>().sqrt();
    let a22_frob = ma_22.iter().map(|x| x * x).sum::<f64>().sqrt();
    let in_cos = (a12_frob / (a11_frob * a22_frob).max(1e-18)).clamp(0.0, 1.0);

    let b12_frob = mb_12.iter().map(|x| x * x).sum::<f64>().sqrt();
    let b11_frob = mb_11.iter().map(|x| x * x).sum::<f64>().sqrt();
    let b22_frob = mb_22.iter().map(|x| x * x).sum::<f64>().sqrt();
    let out_cos = (b12_frob / (b11_frob * b22_frob).max(1e-18)).clamp(0.0, 1.0);

    Ok(AdapterOrthogonalityPair {
        adapter_lhs: lhs.adapter_id.clone(),
        adapter_rhs: rhs.adapter_id.clone(),
        frobenius_cosine: frob_cos,
        input_subspace_cosine: in_cos,
        output_subspace_cosine: out_cos,
        is_nearly_orthogonal: frob_cos <= cosine_threshold,
    })
}

/// Audit all $\binom{K}{2}$ pairs across a list of specialist harness adapters.
pub fn audit_adapter_orthogonality(
    adapters: &[HarnessLoRAAdapter],
    cosine_threshold: f64,
) -> Result<SuperAdapterOrthogonalityAudit, String> {
    if adapters.is_empty() {
        return Err("at least one adapter is required for orthogonality audit".to_string());
    }
    let threshold = cosine_threshold.clamp(1e-6, 1.0);
    let total_rank = adapters.iter().map(|a| a.rank).sum();
    let mut pairs = Vec::new();
    let mut max_cos = 0.0f64;
    let mut sum_cos = 0.0f64;

    for i in 0..adapters.len() {
        for j in (i + 1)..adapters.len() {
            let p = pairwise_adapter_orthogonality(&adapters[i], &adapters[j], threshold)?;
            if p.frobenius_cosine > max_cos {
                max_cos = p.frobenius_cosine;
            }
            sum_cos += p.frobenius_cosine;
            pairs.push(p);
        }
    }

    let mean_cos = if pairs.is_empty() {
        0.0
    } else {
        sum_cos / (pairs.len() as f64)
    };
    let all_orth = pairs.iter().all(|p| p.is_nearly_orthogonal);

    Ok(SuperAdapterOrthogonalityAudit {
        adapter_count: adapters.len(),
        total_concatenated_rank: total_rank,
        cosine_threshold: threshold,
        max_pairwise_cosine: max_cos,
        mean_pairwise_cosine: mean_cos,
        all_pairs_orthogonal: all_orth,
        pairs,
    })
}

/// Compose $K$ specialist harness adapters into a unified Super-Adapter (`Qwen3.8-27B-super-adapter`
/// `combination_type = "cat"` parity, default weight $0.5$ per adapter when `weights` is `None`).
pub fn compose_super_adapter(
    super_adapter_id: impl Into<String>,
    adapters: &[HarnessLoRAAdapter],
    weights: Option<&[f32]>,
    mode: SuperAdapterCombineMode,
    cosine_threshold: f64,
) -> Result<(HarnessLoRAAdapter, SuperAdapterCompositionReceipt), String> {
    if adapters.is_empty() {
        return Err("cannot compose an empty adapter list".to_string());
    }
    let rows = adapters[0].rows;
    let cols = adapters[0].cols;
    let target_tensor = adapters[0].target_tensor.clone();
    for a in adapters.iter().skip(1) {
        if a.rows != rows || a.cols != cols {
            return Err("all adapters in a super-adapter must share rows and cols".to_string());
        }
    }

    let default_w = vec![0.5f32; adapters.len()];
    let w_slice = match weights {
        Some(w) => {
            if w.len() != adapters.len() {
                return Err("weights length must match adapters length".to_string());
            }
            w
        }
        None => &default_w,
    };

    let pre_audit = audit_adapter_orthogonality(adapters, cosine_threshold)?;
    let mut working = adapters.to_vec();

    if mode == SuperAdapterCombineMode::OrthogonalizedRankConcat && !pre_audit.all_pairs_orthogonal
    {
        for i in 1..working.len() {
            let (prior, tail) = working.split_at_mut(i);
            let current = &mut tail[0];
            for prev in prior.iter() {
                current.project_orthogonal_against(prev)?;
            }
        }
    }

    let post_max_cos = if mode == SuperAdapterCombineMode::OrthogonalizedRankConcat {
        audit_adapter_orthogonality(&working, cosine_threshold)?.max_pairwise_cosine
    } else {
        pre_audit.max_pairwise_cosine
    };

    let super_id = super_adapter_id.into();
    let constituent_ids: Vec<String> = working.iter().map(|a| a.adapter_id.clone()).collect();
    let harness_label = working
        .iter()
        .map(|a| a.harness_name.as_str())
        .collect::<Vec<_>>()
        .join("+");

    let composed = match mode {
        SuperAdapterCombineMode::RankConcat | SuperAdapterCombineMode::OrthogonalizedRankConcat => {
            let total_rank: usize = working.iter().map(|a| a.rank).sum();
            let mut super_a = Vec::with_capacity(total_rank * cols);
            let mut super_b = vec![0.0f32; rows * total_rank];

            let mut rank_offset = 0usize;
            for (adapter, &weight) in working.iter().zip(w_slice.iter()) {
                let eff = weight * adapter.scaling();
                let mag = eff.abs().sqrt();
                let sign_a = if eff < 0.0 { -mag } else { mag };
                let scale_b = mag;

                // Append scaled rows of A: (r_i × cols)
                for &v in &adapter.lora_a {
                    super_a.push(v * sign_a);
                }

                // Place scaled columns of B into rows × [rank_offset .. rank_offset + r_i]
                for row in 0..rows {
                    let src = &adapter.lora_b[row * adapter.rank..(row + 1) * adapter.rank];
                    let dst = &mut super_b[row * total_rank + rank_offset
                        ..row * total_rank + rank_offset + adapter.rank];
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d = s * scale_b;
                    }
                }
                rank_offset += adapter.rank;
            }

            // Set lora_alpha = total_rank so scaling() == 1.0 (exact per-adapter weights already absorbed)
            HarnessLoRAAdapter::new(
                super_id.clone(),
                harness_label,
                target_tensor,
                rows,
                cols,
                total_rank,
                total_rank as f32,
                super_a,
                super_b,
            )?
        }
        SuperAdapterCombineMode::LinearAverageBaseline => {
            let r = working[0].rank;
            if working.iter().any(|a| a.rank != r) {
                return Err("LinearAverageBaseline requires equal adapter ranks".to_string());
            }
            let mut avg_a = vec![0.0f32; r * cols];
            let mut avg_b = vec![0.0f32; rows * r];
            for (adapter, &weight) in working.iter().zip(w_slice.iter()) {
                let eff = weight * adapter.scaling();
                let mag = eff.abs().sqrt();
                let sign_a = if eff < 0.0 { -mag } else { mag };
                for (dst, &src) in avg_a.iter_mut().zip(adapter.lora_a.iter()) {
                    *dst += src * sign_a;
                }
                for (dst, &src) in avg_b.iter_mut().zip(adapter.lora_b.iter()) {
                    *dst += src * mag;
                }
            }
            HarnessLoRAAdapter::new(
                super_id.clone(),
                harness_label,
                target_tensor,
                rows,
                cols,
                r,
                r as f32,
                avg_a,
                avg_b,
            )?
        }
    };

    let fp16_bytes = (composed.lora_a.len() + composed.lora_b.len()) * 2;
    let receipt = SuperAdapterCompositionReceipt {
        super_adapter_id: super_id,
        mode,
        constituent_adapters: constituent_ids,
        weights_applied: w_slice.to_vec(),
        resulting_rank: composed.rank,
        pre_audit,
        post_orthogonalization_max_cosine: post_max_cos,
        parameters_bytes_fp16: fp16_bytes,
    };

    Ok((composed, receipt))
}

/// Fuse a Rank-$r_s$ `HarnessLoRAAdapter` (or Rank-Concatenated Super-Adapter) directly onto a
/// sub-1-bit `.jsyn` [`LittleBitSVFTensor`] by appending $r_s$ Dual-SVID binary bitplanes along the
/// latent rank axis (`new_rank = base.rank + adapter.rank`), keeping the entire model in packed
/// `XNOR + POPCNT` format!
pub fn append_super_adapter_to_littlebit(
    base: &LittleBitSVFTensor,
    adapter: &HarnessLoRAAdapter,
) -> Result<LittleBitSVFTensor, String> {
    if base.rows != adapter.rows || base.cols != adapter.cols {
        return Err(format!(
            "LittleBit tensor {}x{} does not match adapter {}x{}",
            base.rows, base.cols, adapter.rows, adapter.cols
        ));
    }

    let new_rank = base
        .rank
        .checked_add(adapter.rank)
        .ok_or_else(|| "combined rank overflow".to_string())?;
    let new_rank_words = new_rank.div_ceil(64);
    let mut new_u_bits = vec![0u64; base.rows * new_rank_words];
    let mut new_v_bits = vec![0u64; base.cols * new_rank_words];
    let mut new_latent_scale = Vec::with_capacity(new_rank);
    new_latent_scale.extend_from_slice(&base.latent_scale);

    // Copy existing base bitplanes [0 .. base.rank)
    for r in 0..base.rows {
        let src = &base.u_bits[r * base.rank_words..(r + 1) * base.rank_words];
        let dst = &mut new_u_bits[r * new_rank_words..r * new_rank_words + base.rank_words];
        dst.copy_from_slice(src);
    }
    for c in 0..base.cols {
        let src = &base.v_bits[c * base.rank_words..(c + 1) * base.rank_words];
        let dst = &mut new_v_bits[c * new_rank_words..c * new_rank_words + base.rank_words];
        dst.copy_from_slice(src);
    }

    // Quantize each adapter rank channel k in [0 .. adapter.rank) normalized by base row/col scales:
    // ΔW_{i,j}^{(k)} = gamma * B_{i,k} * A_{k,j} ≈ row_scale[i] * U_bin[i, base+k] * sigma_{base+k} * V_bin[j, base+k] * col_scale[j]
    let gamma = adapter.scaling();
    for k in 0..adapter.rank {
        let global_k = base.rank + k;
        let word_idx = global_k / 64;
        let bit_mask = 1u64 << (global_k % 64);

        let mut u_norm_sum = 0.0f64;
        for i in 0..base.rows {
            let rs = base.row_scale[i].abs().max(1e-6);
            let b_val = adapter.lora_b[i * adapter.rank + k] / rs;
            if b_val >= 0.0 {
                new_u_bits[i * new_rank_words + word_idx] |= bit_mask;
            }
            u_norm_sum += b_val.abs() as f64;
        }

        let mut v_norm_sum = 0.0f64;
        for j in 0..base.cols {
            let cs = base.col_scale[j].abs().max(1e-6);
            let a_val = adapter.lora_a[k * adapter.cols + j] / cs;
            if a_val >= 0.0 {
                new_v_bits[j * new_rank_words + word_idx] |= bit_mask;
            }
            v_norm_sum += a_val.abs() as f64;
        }

        let mean_u = (u_norm_sum / (base.rows as f64)) as f32;
        let mean_v = (v_norm_sum / (base.cols as f64)) as f32;
        new_latent_scale.push(gamma * mean_u * mean_v);
    }

    // Extend registered SVF expert vectors with neutral `1.0` scales on the newly appended ranks
    let mut new_experts = std::collections::BTreeMap::new();
    for (ename, z) in &base.svf_experts {
        let mut ext = z.clone();
        ext.resize(new_rank, 1.0f32);
        new_experts.insert(ename.clone(), ext);
    }

    Ok(LittleBitSVFTensor {
        name: format!("{}+{}", base.name, adapter.adapter_id),
        rows: base.rows,
        cols: base.cols,
        rank: new_rank,
        rank_words: new_rank_words,
        u_bits: new_u_bits,
        v_bits: new_v_bits,
        row_scale: base.row_scale.clone(),
        col_scale: base.col_scale.clone(),
        latent_scale: new_latent_scale,
        outliers: base.outliers.clone(),
        svf_experts: new_experts,
        reconstruction_cosine: base.reconstruction_cosine,
    })
}

/// Compute group-relative advantages (`AsyncGRPO` normalization) across rollout episodes grouped
/// by `prompt_group_id`: $\hat{A}_i = (R_i - \mu_G) / (\sigma_G + 10^{-8})$.
pub fn compute_grpo_group_advantages(episodes: &mut [HarnessTrajectoryEpisode]) {
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, ep) in episodes.iter().enumerate() {
        groups
            .entry(ep.prompt_group_id.clone())
            .or_default()
            .push(idx);
    }

    for indices in groups.values() {
        if indices.len() <= 1 {
            if let Some(&i) = indices.first() {
                episodes[i].grpo_advantage = 0.0;
            }
            continue;
        }
        let n = indices.len() as f64;
        let mean = indices.iter().map(|&i| episodes[i].raw_reward).sum::<f64>() / n;
        let var = indices
            .iter()
            .map(|&i| {
                let d = episodes[i].raw_reward - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        if std < 1e-8 {
            for &i in indices {
                episodes[i].grpo_advantage = 0.0;
            }
        } else {
            for &i in indices {
                episodes[i].grpo_advantage = (episodes[i].raw_reward - mean) / (std + 1e-8);
            }
        }
    }
}

/// Export normalized `HarnessTrajectoryEpisode`s as newline-delimited JSON (`JSONL`) compatible
/// with Hugging Face `TRL` / `harness-hub` offline & `AsyncGRPO` replay buffers.
pub fn export_jsonl_for_trl(episodes: &[HarnessTrajectoryEpisode]) -> Result<String, String> {
    let mut lines = Vec::with_capacity(episodes.len());
    for ep in episodes {
        let line = serde_json::to_string(ep).map_err(|e| e.to_string())?;
        lines.push(line);
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pi_opencode_hermes_orthogonal_rank_concat_super_adapter() {
        let rows = 128;
        let cols = 128;
        let pi = HarnessLoRAAdapter::initialized_for_harness(
            "pi-r32",
            "pi",
            "layers.0.q_proj",
            rows,
            cols,
            32,
            64.0,
            101,
        )
        .expect("pi");
        let opencode = HarnessLoRAAdapter::initialized_for_harness(
            "opencode-r32",
            "opencode",
            "layers.0.q_proj",
            rows,
            cols,
            32,
            64.0,
            202,
        )
        .expect("opencode");
        let hermes = HarnessLoRAAdapter::initialized_for_harness(
            "hermes-r32",
            "hermes",
            "layers.0.q_proj",
            rows,
            cols,
            32,
            64.0,
            303,
        )
        .expect("hermes");

        let adapters = vec![pi.clone(), opencode.clone(), hermes.clone()];
        let (super_adapter, receipt) = compose_super_adapter(
            "qwen-super-r96",
            &adapters,
            None, // default 0.5 scaling per harness-hub Qwen3.8-27B-super-adapter
            SuperAdapterCombineMode::RankConcat,
            0.15,
        )
        .expect("compose super adapter");

        assert_eq!(super_adapter.rank, 96);
        assert_eq!(receipt.resulting_rank, 96);
        assert!(
            receipt.pre_audit.max_pairwise_cosine < 0.12,
            "independent harness subspaces should be nearly orthogonal: got {}",
            receipt.pre_audit.max_pairwise_cosine
        );

        // Verify mathematical exactness: super_adapter.forward_delta(x) == 0.5 * sum_i adapter_i.forward_delta(x)
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.13).sin()).collect();
        let y_super = super_adapter.forward_delta(&x).expect("super forward");
        let y_pi = pi.forward_delta(&x).expect("pi fwd");
        let y_oc = opencode.forward_delta(&x).expect("oc fwd");
        let y_he = hermes.forward_delta(&x).expect("he fwd");

        for idx in 0..rows {
            let expected = 0.5 * (y_pi[idx] + y_oc[idx] + y_he[idx]);
            assert!(
                (y_super[idx] - expected).abs() < 1e-5,
                "mismatch at row {idx}: got {} vs expected {expected}",
                y_super[idx]
            );
        }
    }

    #[test]
    fn test_gram_schmidt_deinterference_and_littlebit_bitplane_fusion() {
        let rows = 64;
        let cols = 64;
        let a1 = HarnessLoRAAdapter::initialized_for_harness(
            "sql-r16", "jouledb", "w0", rows, cols, 16, 32.0, 777,
        )
        .expect("a1");
        // Deliberately clone a1 so cosine == 1.0 (worst-case interference!)
        let mut a2 = a1.clone();
        a2.adapter_id = "colliding-r16".to_string();
        a2.harness_name = "kiro".to_string();

        let pre = pairwise_adapter_orthogonality(&a1, &a2, 0.05).expect("pre");
        assert!((pre.frobenius_cosine - 1.0).abs() < 1e-5);
        assert!(!pre.is_nearly_orthogonal);

        let (super_orth, rec) = compose_super_adapter(
            "deinterfered-r32",
            &[a1.clone(), a2],
            Some(&[1.0, 1.0]),
            SuperAdapterCombineMode::OrthogonalizedRankConcat,
            0.05,
        )
        .expect("orth compose");
        assert!(rec.post_orthogonalization_max_cosine < 1e-4);

        // Fuse onto a sub-1-bit `.jsyn` LittleBitSVFTensor
        let dense: Vec<f32> = (0..(rows * cols))
            .map(|k| ((k as f32) * 0.07).cos() * 0.2)
            .collect();
        let lb = LittleBitSVFTensor::factorize("w0", &dense, rows, cols, 16, 4, 0).expect("lb");
        let fused = append_super_adapter_to_littlebit(&lb, &super_orth).expect("fuse");
        assert_eq!(fused.rank, 16 + 32);

        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.1).sin()).collect();
        let out = fused.forward_vec(&x, &[]).expect("fused fwd");
        assert_eq!(out.len(), rows);
    }

    #[test]
    fn test_grpo_group_advantages_and_online_step() {
        let mut episodes = vec![
            HarnessTrajectoryEpisode {
                episode_id: "ep1".into(),
                prompt_group_id: "g1".into(),
                harness_name: "opencode".into(),
                prompt: "SELECT balance".into(),
                steps: vec![],
                final_answer: "ok".into(),
                raw_reward: 1.0,
                grpo_advantage: 0.0,
            },
            HarnessTrajectoryEpisode {
                episode_id: "ep2".into(),
                prompt_group_id: "g1".into(),
                harness_name: "opencode".into(),
                prompt: "SELECT balance".into(),
                steps: vec![],
                final_answer: "bad".into(),
                raw_reward: -1.0,
                grpo_advantage: 0.0,
            },
        ];
        compute_grpo_group_advantages(&mut episodes);
        assert!(episodes[0].grpo_advantage > 0.99);
        assert!(episodes[1].grpo_advantage < -0.99);

        let mut adapter = HarnessLoRAAdapter::initialized_for_harness(
            "oc-r8", "opencode", "q", 16, 16, 8, 16.0, 42,
        )
        .expect("adapter");
        let x = vec![0.25f32; 16];
        let target = vec![0.10f32; 16];
        let mse0 = adapter
            .grpo_advantage_step(&x, &target, episodes[0].grpo_advantage as f32, 0.08)
            .expect("step0");
        let mse1 = adapter
            .grpo_advantage_step(&x, &target, episodes[0].grpo_advantage as f32, 0.08)
            .expect("step1");
        assert!(
            mse1 < mse0,
            "positive GRPO advantage should reduce target MSE"
        );
    }
}
