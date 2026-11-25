# The `.jsyn` (`JSYN-v1`) Living-Weights Specification & OpenIE Model Ecosystem

**Version:** `1.2` (`JouleDB v0.6.0` Pre-1.0 Research Preview — November 2026)  
**License:** `Apache-2.0 OR MIT` (per-model `.jsyn` exports preserve exact upstream SPDX / model licenses)  
**Public Model on Hugging Face:** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`Apache-2.0`, `0.44 BPW`, `1.95 MB`, measured sentence-embedding cosine `0.9441` vs upstream `FP32`; 81 unverified low-BPW causal LLM exports retracted in `v0.6.0`)  
**Reference Implementation:** `crates/joule-db-weights/src/{synapse.rs, checkpoint_ingest.rs, causal_decoder.rs, rank_sweep_bench.rs}`

---

## 1. Why `.jsyn` Exists: Models Belong Inside the Database, Not in Frozen Slabs

Modern AI infrastructure inherited a broken fifty-year separation between **passive databases** and **frozen neural weight files** (`.safetensors`, `.gguf`, `.onnx`):

1. **Dense FFN Matrices Waste ~85% of Parameters Acting as a Lossy Key-Value Store:** Because standalone LLMs have no transactional database behind them, they must memorize static world facts across thousands of low-energy tail singular values in feed-forward matrices—burning tens of gigabytes of VRAM and memory bandwidth on every token.
2. **Scalar Quantization Hits a Hard `1.0 Bit/Weight` (`16x`) Wall:** Entry-by-entry quantizers (`Q4_K_M`, `NVFP4`, `FP8`, `BitNet b1.58`, `MLX 1-bit`) store at least 1 bit per $(i, j)$ matrix coordinate ($m \cdot n$ bits), leaving multi-billion-parameter checkpoints too heavy for instant sandboxes, edge robotics, or zero-cold-start Kubernetes routing.
3. **Frozen Weights Cannot Adapt or Roll Back Atomically:** Correcting a stale fact or adapting to a domain in `.safetensors`/`.gguf` requires external fine-tuning pipelines or context-window prompt bloat, with zero MVCC LSN rollback if live updates poison behavior.

**JouleDB + `.jsyn` (`JouleSynapse Living Weights`)** resolves all three by splitting **neural circuitry** from **stored knowledge** inside one ACID/SSI thermodynamic database:

- **Exact facts, documents, relationships, and time-series** live in MVCC SQL tables, `Engram` $O(1)$ conditional memory (`220 nJ`), `TriModal RRF` (`BM25 ⋈ RaBitQ 1-bit POPCNT ⋈ HDC`), and `PathHD` (`70.0%` WN18RR `Hits@1` at `10 µJ/pred`, `100%` `LongMemEval-60`).
- **Neural representations, syntax, routing, embeddings, and multimodal encoders** live in `.jsyn` sub-1-bit bitplanes (`0.02–0.50 BPW`, **`20x–560x` smaller than `FP16`**)—queryable via SQL (`SELECT * FROM synapse_tensors`), Copy-on-Write branchable in $O(1)$, online-plastic via **Sakana `SVF`** and **Google `Titans`**, and metered across 6 hardware energy rails (`nJ`).

---

## 2. Binary Envelope (`JSYN-v1`)

Every `.jsyn` snapshot is written atomically (`write(.tmp) -> fsync -> rename -> dir fsync`) and verified by IEEE `CRC-32` before decompression (with a hard `2 GiB` 64-bit decompression ceiling against corrupt or hostile inputs):

```text
+-------------------+-------------------+-------------------+---------------------------+-------------------------------+
| Magic: b"JSYN"    | Version: u32 (1)  | IEEE CRC-32: u32  | Raw Payload Len: u64 (LE) | zstd (level 3) Payload Stream |
| 4 bytes           | 4 bytes           | 4 bytes           | 8 bytes                   | variable                      |
+-------------------+-------------------+-------------------+---------------------------+-------------------------------+
```

### Internal Payload Sections

1. **`littlebit_tensors` (`BTreeMap<String, LittleBitSVFTensor>`):**
   - `rows: usize`, `cols: usize`, `rank: usize`, `rank_words: usize` (`ceil(rank / 64)`)
   - `u_bits: Vec<u64>`, `v_bits: Vec<u64>`: FastLanes-packed binary sign factors ($\{-1, +1\}$)
   - `row_scale: Vec<f32>`, `col_scale: Vec<f32>`, `latent_scale: Vec<f32>`: continuous multi-scale channel & singular gains
   - `outliers: Vec<ResidualOutlier>`: sparse heavy-tailed super-weight coordinates `(row: u32, col: u32, delta: f32)`
   - `svf_experts: BTreeMap<String, Vec<f32>>`: named **Sakana `Transformer²` SVF** singular scaling vectors $z \in \mathbb{R}^r$
2. **`vectors_1d` (`BTreeMap<String, Vec<f32>>` — 0% Lossy Quantization):**
   - Preserves all sensitive 1D structural tensors in **exact `FP32`**: `RMSNorm`, `LayerNorm`, `+1` Gemma offsets, per-head `QK-Norm`, attention/MLP biases, `Liquid AI ShortConv` depthwise `1x3` kernels, and `Mamba-2` `A_log`, `D`, `dt_bias`, and `conv1d` parameters.
3. **`model_configs` (`BTreeMap<String, CausalModelMeta>`):**
   - Self-contained execution graph metadata (`num_attention_heads`, `num_key_value_heads`, `head_dim`, `rope_theta`, `rms_norm_eps`, `hidden_act`, `model_type`, MoE routing parameters, `conv_l_cache`).
4. **`model_vocabs` & `model_bpe_merges`:**
   - Embedded GPT-2 byte-level BPE, SentencePiece `▁` Metaspace BPE, or WordPiece vocabulary so `.jsyn` runs with zero Python `transformers` or `tokenizers` dependencies.
5. **`engram` (`EngramMemoryTable`):**
   - Canonicalized multi-head hash-addressed conditional memory entries for $O(1)$ factual grounding.

---

## 3. Mathematical Formulation & Zero-Decompression Execution

Given a 2D weight projection $W \in \mathbb{R}^{m \times n}$ (or a 3D stacked Mixture-of-Experts tensor $[E, R, C]$ folded to $[E \cdot R, C]$), `LittleBit-2 ⊕ Sakana SVF` approximates $W$ as:

$$W \approx \operatorname{diag}(s_{\text{row}}) \cdot \left( \sum_{k=1}^{r} \sigma_k z_k \, u_k v_k^\top \right) \cdot \operatorname{diag}(s_{\text{col}}) + S_{\text{outlier}}$$

where $u_k \in \{-1, +1\}^m$ and $v_k \in \{-1, +1\}^n$.

### Effective Bits-Per-Weight (`BPW`)

$$\text{BPW}(m, n, r, K) = \frac{(m + n)\,r + 16\,(m + n + r) + 48\,K}{m \cdot n}$$

For a standard `4096 x 4096` projection (`33.55 MB` in `FP16`):
- **`r = 8`, `K = 32`:** `0.0118 BPW` (`1,353x` raw matrix compression)
- **`r = 32`, `K = 64`:** `0.0238 BPW` (`672x` raw matrix compression)
- **`r = 64`, `K = 64`:** `0.0394 BPW` (`406x` raw matrix compression)
- **`r = 128`, `K = 64`:** `0.0706 BPW` (`226x` raw matrix compression)

### Factored Forward Pass ($O((m + n) \cdot r)$ Without `FP16` Materialization)

Evaluating $y = \hat{W}x$ never reconstructs $W$ in memory:
1. Input channel scaling: $\tilde{x}_j = s_{\text{col}, j} x_j$ in $O(n)$.
2. Latent binary projection via L1-resident **`T-MAC` 256-entry byte LUTs** (CPU) or `wgpu` **`countOneBits` (`XNOR-POPCNT`)** compute shaders (Metal/Vulkan/WebGPU) in $O(n \cdot r)$:
   $$\alpha_k = \sigma_k z_k \sum_{j=1}^{n} v_{j,k}\,\tilde{x}_j$$
3. Output binary projection + $K$ sparse outliers in $O(m \cdot r + K)$:
   $$y_i = s_{\text{row}, i} \sum_{k=1}^{r} u_{i,k}\,\alpha_k + \sum_{(i,j,\Delta) \in S_{\text{outlier}}} \Delta \cdot x_j$$

For 128-expert Mixture-of-Experts models (`Inkling-2-Mini`, `TML-d24-VarLoop`, `Gemma-4-26B-A4B`, `Nemotron-3-Nano-30B-A3B`, `Poolside Laguna-XS-2.1`), `forward_row_slice` evaluates **only the active top-$k$ expert row slices** in $O(k_{\text{active}} \cdot R \cdot r)$.

---

## 4. Empirical-Bayes `AWQ ⊕ GPTQ` Activation-Aware `SVF` Calibration

Real Transformer activations exhibit extreme channel skew (`~1–2%` of channels carry `8x–14x` higher magnitude). `POST /api/v1/synapse/littlebit/calibrate` aligns `latent_scale` and `row_scale` directly against empirical calibration activations $X_{\text{calib}}$ using alternating Gauss-Seidel coordinate-ridge updates with sample-size-aware Empirical-Bayes shrinkage toward `1.0`:

- **`0` extra storage bits** (`base_bitplanes_mutated = false`),
- **`92.04%` skewed-activation MSE reduction** (`14.25 -> 1.13` on Apple M5 Max), and
- Strictly non-overfitting on held-out `X_eval` even when `calib_samples < rank` (verified by `POST /api/v1/synapse/bench/rank_sweep`).

---

## 5. Transparent Information-Theoretic Boundary & Measured Fidelity

- **Where Sub-1-Bit `.jsyn` Works Well (`>= 0.40 BPW` Encoders):** Bidirectional sentence embeddings (`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit` at `0.4378 BPW` / `1.95 MB`, achieving **`0.9441` sentence-embedding cosine similarity** against upstream `FP32` `sentence-transformers/all-MiniLM-L6-v2`), semantic routing, and retrieval candidate generation.
- **Where Extreme Low-Rank (`0.02–0.17 BPW`, `r = 8..128`) Degrades Severely:** On multi-billion-parameter causal LLMs (`Gemma-4-E2B`, `Inkling-2-Mini`, `SmolLM2-135M`), out-of-sample rank sweeps (`POST /api/v1/synapse/bench/rank_sweep`) show held-out output cosine of **`0.271–0.455`**, Softmax `KL = 2.72–3.87`, and **`65.6%` Top-5 logit agreement** at `r = 128` (`0.166 BPW`). At that fidelity, `.jsyn` causal checkpoints **do not function as drop-in replacements for the upstream models**, and speculative drafting is only lossless when verified against an unquantized dense target model (not self-verified against a low-rank `.jsyn` target). For this reason, OpenIE retracted the 81 unverified low-BPW causal LLM exports from Hugging Face in `v0.6.0`.

---

## 6. Supported Checkpoint Ingest Formats & Experimental Operator Coverage

`checkpoint_ingest` compiles checkpoints directly via zero-copy `mmap`:
- **`.safetensors`:** `F32`, `F16`, `BF16`, scaled `FP8` (`E4M3`/`E5M2`), NVIDIA Blackwell / Thinking Machines `NVFP4` (`E2M1` + `F8_E4M3` block scale + `weight_scale_2`), MLX packed affine `U32` (`1/2/4/8`-bit), and 3D stacked MoE `[E, R, C]`.
- **`.gguf` (v2/v3):** `Q1_0_G128`, `Q1_0`, `I2_S` (`BitNet.cpp`), `Q2_0`/`PQ2_0`/`PTQ1_0` (`PrismML Ternary-Bonsai-2`), `TQ1_0`, `TQ2_0`, `Q2_K`..`Q6_K`, `Q4_0`..`Q8_0`, `BF16`, `F16`, `F32` + embedded tokenizer extraction.
- **Architectures:** `Llama-3`, `Qwen2.5`/`Qwen3`/`Qwen3-MoE`/`Qwen3.5` (Gated DeltaNet + fused Q-gate), `Google Gemma 4` (`+1` RMSNorm, `GeGLU`, `attention_k_eq_v`, shared KV), `Thinking Machines Inkling-2 & TML-d24-VarLoop`, `NVIDIA Nemotron-H` (`Mamba-2` + `GQA`), `Liquid AI LFM2/2.5/3` (`LIV ShortConv`), `PrismML Bonsai & Ternary-Bonsai-2`, `Poolside Laguna-XS-2.1`, `Microsoft BitNet b1.58 & Aion`, `TII Falcon-H1`, `Zyphra Zamba2 & ZONOS3`, `AllenAI OLMo-2`, `Cactus Needle`, `IBM Granite 4.1 & ModernBERT`, `SmolVLM/SmolVLA`, `SigLIP`, and `Whisper`.
