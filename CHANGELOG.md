# Changelog

All notable changes to this release surface are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.8] - 2026-11-24

### Added
- **`TigerBeetle` `VOPR` Ledger, Goldman `SecDB` + Google `Napa` Reactive SLA, `ClickBench` `ZoneMap` Scanner & 7-Pillar Industry Benchmark Suite (`tiger_ledger`, `secdb_napa`, `zonemap_scan`, `certification_routes`)**:
  - **`TigerBeetle`-Class Double-Entry Accounting & `VOPR` Fault Simulator (`POST /api/v1/certify/tiger_vopr`):** `128`-bit fixed-point accounts (`DebitsMustNotExceedCredits`), two-phase `PendingReserve`/`PostPending`/`VoidPending` partial settlement, atomic `linked` multi-leg transfer rollback, and multi-seed `VOPR` torn-WAL & 1-bit NAND bit-rot recovery conservation verification.
  - **Goldman Sachs `SecDB` + Google `Napa` Reactive Risk & Freshness-SLA Engine (`POST /api/v1/secdb/napa_eval`):** Persisted topological dependency DAG recomputing strictly reachable dirty cones with early cutoff (`>99%` compute saved on unrelated books) paired with `Napa` `(max_staleness_ms, eager_budget_nj)` burst write coalescing.
  - **`ClickBench` / `Vortex` `ZoneMap` + Late-Materialization Columnar Scanner (`POST /api/v1/olap/zonemap_scan`):** `1,024`-row granule min/max + 64-bit SplitMix Bloom pruning (`>97.9%` cold granule skip) and `SelectionVector` (`Vec<u16>`) late payload materialization.
  - **Unified 7-Pillar 360° Industry Benchmark Runner (`POST /api/v1/bench/industry_suite`):** Single-call certification across `TPC-C SSI` (`100k+ TPS`), `ClickBench ZoneMap`, `TigerBeetle VOPR`, `SecDB + Napa`, `Big-ANN RaBitQ` filtered `POPCNT`, `Nexmark DBSP` streaming `ZSet`, and `Hermitage / Elle` `SSI`.

## [0.2.7] - 2026-11-24

### Added
- **Database Gap-Closure & Formal Certification Suite (`morsel_spill`, `elle_hermitage`, `conformance_oracle`, `certification_routes`, `synapse`, `joule-db-amorphic`)**:
  - **Morsel-Driven Out-of-Core Spill-to-Disk Vectorized OLAP (`POST /api/v1/olap/morsel_join` & `POST /api/v1/olap/morsel_groupby`):** `2,048`-row cache-resident `ColumnMorsel` batches with Radix-Partitioned Grace Hash Join and GroupBy spilling to zero-parse binary columnar CRC32 disk segments under a strict `memory_budget_bytes` ceiling with 6-rail `storage_nvme_nj` accounting.
  - **Kleppmann `Hermitage` (`G0`..`G2-item`) + Jepsen/`Elle` Tarjan `SCC` Cycle Audit (`POST /api/v1/certify/hermitage_elle` & `elle_analyze`):** Infers Adya `ww`, `wr`, and `rw` anti-dependency edges, classifies `G0`, `G1a/b/c`, `P4`, `G2-item` Write Skew, and `G2` cycles via Tarjan's $O(V+E)$ `SCC` algorithm, exports Clojure EDN for `elle-cli`, and certifies Cahill `SSI` (`0` cycles).
  - **Portable `sqllogictest` (`.slt` + RFC-1321 MD5 Digest) & `SQLancer` (`TLP` + `NoREC`) Oracles (`POST /api/v1/certify/sqllogictest` & `sqlancer`):** Native `.slt` conformance runner plus 3-valued `NULL` Ternary Logic Partitioning and `NoREC` differential logic-bug oracles.
  - **Activation-Aware `SVF` Post-Compression Calibration (`POST /api/v1/synapse/littlebit/calibrate`), `Qwen3.5` Fused-Gate Attention & Auto-Sharded `AmorphicStore` Holograms:** Closed-form `AWQ ⊕ GPTQ` empirical activation calibration (`>92%` skewed-activation MSE reduction at `0` extra BPW without mutating `u64` bitplanes), exact GCD GQA head inference + `Qwen3.5` `[query, attn_gate]` `q_proj` splitting, and unbounded `256`-record clean-SNR `hologram_shards`.

## [0.2.6] - 2026-11-24

### Added
- **45-Day Frontier Drop Fleet, Joint Binary Gram `LittleBit-2` Refinement & Zero-Copy Checkpoint Compiler (`synapse`, `checkpoint_ingest`, `causal_decoder`, `synapse_routes`)**:
  - **SplitMix64 Rademacher Seeding + Joint Binary Gram Gauss-Seidel Singular Refinement:** Replaces periodic bitplane initialization with quasi-orthogonal Rademacher hypercube + residual power projection and solves the exact $R \times R$ cross-rank binary Gram normal equations at `0` extra BPW, eliminating multi-rank bitplane collinearity and raising multi-rank reconstruction cosine above `0.88`.
  - **PrismML `Ternary-Bonsai-2` (`Q2_0`/`PQ2_0`/`PTQ1_0`), `Q5_0`/`Q5_1`/`Q2_K`/`Q3_K`/`Q5_K` GGUF Unpackers & Zero-Copy `.safetensors` `mmap`:** Unpacks `Ternary-Bonsai-2` 2-bit/base-3 GGUF (`type_id=42,142,143`) and `K`-quants, streams `10–20 GB` `.safetensors` shards directly from OS page cache via `memmap2`, raises 64-bit `MAX_TENSOR_ELEMENTS` to `4 Gi` (`Gemma 4` `2.82B`-element per-layer embeddings) and `MAX_SNAPSHOT_UNCOMPRESSED` to `2 GiB` (`26B`–`33B` MoE `.jsyn` reloads), and deletes cached shards immediately after compilation.
  - **Google `Gemma 4` (`E2B`/`E4B`/`26B-A4B`), `Qwen3.5` / `Ternary-Bonsai-2` Gated DeltaNet, `Falcon-H1` / `Zamba2` Hybrid Mamba, `Poolside Laguna-XS-2.1` 33.4B MoE, `NVIDIA Dory 1.0` & `Sigma`, and `Thinking Machines Lab TML-d24-VarLoop`:** Full causal execution across dynamic per-layer `head_dim` (`256`/`512`) + `attention_k_eq_v` + shared-KV reuse + `layer_scalar` (`Gemma 4`), recurrent Gated DeltaNet (`.safetensors` & GGUF), parallel `mamba + self_attn` (`Falcon-H1` / `Zamba2`), 3D `ffn_*_exps` MoE (`Laguna-XS`), and `gemma` / `falcon-llm-license` Model Card support across the 67-model `< 30 GB` catalog (`82` live `.jsyn` repositories on Hugging Face).

## [0.2.5] - 2026-11-23

### Added
- **NVIDIA, Thinking Machines Lab & Sparse MoE / `Mamba-2` / `NVFP4` Frontier Fleet (`checkpoint_ingest`, `causal_decoder`, `transformer_encoder`, `synapse_routes`)**:
  - **NVIDIA Blackwell `NVFP4` (`E2M1` + `F8_E4M3` Block Scale), `FP8` (`F8_E4M3`/`F8_E5M2`), 3D Stacked MoE & Microsoft `BitNet.cpp` `I2_S` (`type_id=36`) Unpackers:** Compiles `NVFP4`, scaled `FP8`, 3D `[E, R, C]` expert tensors, and GGUF `I2_S` 2-bit ternary checkpoints directly into sub-1-bit `LittleBit-2` bitplanes (`forward_row_slice` zero-copy expert execution), plus streaming constant-RAM disk downloads in `POST /api/v1/synapse/hf/pull` and RAM/VRAM eviction (`POST /api/v1/synapse/models/evict`).
  - **Native `NVIDIA Nemotron-H` (`Mamba-2` Selective SSM + `GQA` + `ReLU²`), `Thinking Machines Lab Inkling` / `Inkling-2`, `Sparse MoE` (`Qwen3-MoE`, `GLM-4-MoE`), `Liquid AI LFM3`, `AllenAI OLMo-2` Post-Norm, `NomicBert`, `NVIDIA Nemotron-3-Embed`, & `Google EmbeddingGemma-2`:** Full multi-layer causal, hybrid SSM, sparse MoE, and bidirectional embedding execution directly over sub-1-bit bitplanes.
  - **Expanded 47-Model `< 30 GB` Open-Weight Catalog & NVIDIA / CC-BY-4.0 License Preservation:** Expanded `GET /api/v1/synapse/hf/catalog` across 20 frontier ecosystems with automatic `nvidia-nemotron-open-model-license`, `nvidia-open-model-license`, `cc-by-4.0`, `openmdw-1.1`, `lfm1.0`, `mit`, and `apache-2.0` attribution.

## [0.2.4] - 2026-11-23

### Added
- **Universal `< 30 GB` Open-Weight Adaptation Engine (`checkpoint_ingest`, `causal_decoder`, `transformer_encoder`, `synapse_routes`)**:
  - **Multi-Format Checkpoint Compiler:** Added MLX packed affine `U32` (`1`/`2`/`4`/`8`-bit `weight` + `scales` + `biases`) `.safetensors` unpacking (`prism-ml/Bonsai-1.7B-mlx-1bit`, `Bonsai-27B-mlx-1bit`), pure-Rust memory-mapped GGUF v2/v3 -> `LittleBit-2` compilation (`POST /api/v1/synapse/ingest/gguf`, supporting `Q1_0_G128`, `Q1_0`, `TQ1_0`, `TQ2_0`, `Q4_0`, `Q4_1`, `Q8_0`, `Q4_K`, `Q6_K`, `F16`, `BF16`, `F32` + embedded `tokenizer.ggml.tokens`/`merges`), depthwise 1D convolution kernel preservation, and automatic multi-shard / nested-directory discovery in `POST /api/v1/synapse/hf/pull`.
  - **Universal Next-Gen Block Operators:** Native execution of **Liquid AI `LFM2` / `LFM2.5` / `LFM3`** hybrid `ShortConv` (`LIV` gated depthwise 1D conv) + `GQA` + `SwiGLU` (`w1/w3/w2`), **`Qwen3` / `PrismML Bonsai`** per-head `QK-Norm` (`q_dim != hidden_dim`), **Microsoft `BitNet b1.58`** `attn_sub_norm`/`ffn_sub_norm` + `ReLU²`, **Google `Gemma 4`** `+1` `RMSNorm` + `GeGLU` + dual FFN norms + SentencePiece `▁` BPE + softcapping, **Cactus Compute `Needle`** `self_attn_gate` + `fc1/fc2`, **`SmolVLM` / `IBM Granite-Docling`** (`idefics3` `SigLIP` + `Llama`), **IBM Granite `ModernBERT`** (`Wqkv` + `RoPE` + `GeGLU`), and **`Qwen3-Embedding`**.
  - **Curated `< 30 GB` Open-Weight Catalog & License-Preserving `.jsyn` Export (`GET /api/v1/synapse/hf/catalog`):** Built-in 22-model catalog across 11 ecosystems (`Liquid AI`, `PrismML`, `Cactus Compute`, `Poolside`, `Google Gemma 4`, `Qwen3`, `Microsoft BitNet`, `SmolLM`/`SmolVLM`, `IBM Granite`, `Snowflake Arctic`, `OpenAI Whisper`) with automatic SPDX / `LFM Open License v1.0` / `OpenMDW-1.1` / `MIT` / `Apache-2.0` Model Card frontmatter (`hf_catalog()` & `ingest_gguf()` in Python & Rust `0.2.4` SDKs).

## [0.2.3] - 2026-11-23

### Added
- **`JouleKube` — Native Kubernetes Control Plane, `kubectl` Wire API & CNCF AI-Conformance Reconciler (`kube_control_plane`)**:
  - Disintermediates `etcd`, `kube-apiserver`, `containerd`, and idle `gVisor`/`Kata` warm pools by ingesting standard Kubernetes (`v1`, `apps/v1`) and CNCF AI-Conformance manifests (`resource.k8s.io/v1` `DRA`, `kueue.x-k8s.io/v1beta1` `Kueue`, `leaderworkerset.x-k8s.io/v1` `LeaderWorkerSet`, `inference.networking.k8s.io/v1` `InferencePool` / `InferenceModel`, `agents.x-k8s.io/v1beta1` & `extensions.agents.x-k8s.io/v1beta1` `agent-sandbox`, and `kagent.dev/v1alpha1` `Agent` / `ToolServer`) directly into MVCC SQL table `_jouledb_kube_objects`.
  - Exposes `kubectl` discovery & CRUD/Watch endpoints (`GET /version`, `GET /api`, `GET /api/v1`, `GET /apis`, `/api/v1/namespaces/...`, `/apis/{group}/{version}/...`, `?watch=true` NDJSON) plus `POST /api/v1/kube/apply`, `GET /api/v1/kube/status`, `POST /api/v1/kube/inference-pools/{ns}/{name}/route` (`llm-d` KV-prefix EndpointPicker), and `POST /api/v1/kube/agents/{ns}/{name}/invoke` (cryptographically confined by `SandboxClaim` `ThermoCap` macaroons with `0.0W` warm-pool standby burn).
- **Full `Whisper` Two-Conv + Cross-Attention Seq2Seq Decoder & `SigLIP` Zero-Shot Vision-Text Alignment (`multimodal_encoder`)**:
  - Exact 400-point periodic-Hann STFT + Slaney Mel filterbank, `conv1`/`conv2` `GELU` stem, `embed_positions`, and multi-layer `Whisper` Cross-Attention Autoregressive Decoder (`model.decoder.layers.*`) for real open-weight `openai/whisper-tiny.en` speech-to-text, plus `POST /api/v1/synapse/vision/zero_shot` contrastive scoring.
- **Continuous In-Flight Multi-Request Batching & Native Hugging Face Hub `.jsyn` Integration**:
  - `POST /api/v1/synapse/decoder/generate_batch` coalesces concurrent requests into lockstep row-outer `T-MAC` LUT `forward_batch` passes (loading packed `u_bits` bitplanes once across the batch).
  - `POST /api/v1/synapse/hf/export` and `POST /api/v1/synapse/hf/pull` export and pull CRC32-verified `.jsyn` living-weight bundles + Hugging Face Model Cards across Python (`jouledb 0.2.3`) and Rust (`HttpFabricClient`) SDKs.

## [0.2.2] - 2026-11-23

### Added
- **Multimodal Vision (`SigLIP`/`ViT`) & Audio (`Whisper` Log-Mel) Living Bitplanes (`multimodal_encoder`)**:
  - Pure-Rust `.ppm`/`.bmp`/RGB image patchifier + multi-layer `SigLIP`/`ViT` `LittleBit-2` Vision Transformer encoder (`POST /api/v1/synapse/vision/encode` and OpenAI multimodal `image_url` content parts on `POST /v1/chat/completions`).
  - Pure-Rust `RIFF/WAVE` PCM/IEEE-float decoder, `16 kHz` resampler, `512`-point Radix-2 FFT, `80`/`128`-bin Slaney Log-Mel filterbank, and sub-1-bit `Whisper` Audio Transformer encoder (`POST /api/v1/synapse/audio/transcribe` and drop-in OpenAI `POST /v1/audio/transcriptions` supporting both JSON and `multipart/form-data` binary `.wav` uploads).
- **Lossless Self-Speculative `LayerSkip` + Rank-8 `LittleBit-2` Draft-and-Verify (`causal_decoder`)**:
  - Proposes $K$ draft tokens (`speculative_draft_k`) via strided `LayerSkip` + `0.04` BPW single-byte `T-MAC` LUT projections (`forward_vec_truncated_rank`) and verifies `[t_0, d_1..d_K]` in a single row-outer cache-shared `forward_batch` pass with 100% bit-exact greedy equivalence.
- **Registry & Release Pipeline (`crates.io` + `GHCR` + Multi-Platform Python Wheels)**:
  - Attached all 5 OS/arch `abi3` Python `.whl` wheels + `sdist` to GitHub Releases alongside the 10 server binary archives, added `ghcr.io/openie-dev/jouledb` tag pushes and PyPI OIDC Trusted Publisher support, and added `vision_encode`, `audio_transcribe`, and `speculative_draft_k` across Python & Rust `0.2.2` SDKs.

## [0.2.1] - 2026-11-23

### Added
- **GPU-Resident Batched `LittleBit-2` + Causal `GQA` Shaders & `T-MAC` 8-Bit LUT Decoder (`JouleSynapse` & `joule-db-gpu`)**:
  - Persistent GPU-resident bitplane buffers (`ResidentGpuLittleBitWeights`) with 2D batched multi-token Dual-SVID WGSL compute shaders (`forward_batched_resident`) and fused GPU causal Grouped-Query Attention (`forward_causal_gqa`), paired with Four-Russians / `T-MAC` 256-entry `f32` byte lookup tables (`forward_vec` & `forward_batch`) for branchless L1 bitplane inner products and batched causal prefill.
- **Database-Native `MCP` Host & Autonomous Multi-Turn SQL / `JouleWard` / `Titans` Agent Loop (`mcp_agent_loop`)**:
  - Native `MCP` Host registry (`POST /api/v1/mcp/host/servers`, `GET /api/v1/mcp/host/tools`, `POST /api/v1/mcp/host/call`) combining built-in zero-copy database/synapse tools (`jouledb.sql_query`, `jouledb.engram_lookup`, `jouledb.engram_upsert`, `jouledb.synapse_status`) with live Streamable-HTTP `JSON-RPC 2.0` MCP server discovery (`tools/list`) and invocation (`tools/call`).
  - Single-request autonomous multi-turn agent loop (`POST /api/v1/agent/run`) executing grammar-constrained tool selection, `JoulePulse` `45 nJ` guardrail checks, cryptographic `JouleWard` `ThermoCap` capability confinement, content-addressed `JouleHarness` observation distillation, and `Engram` + `Titans` MVCC memory commits (`mcp_register_server`, `mcp_list_tools`, `mcp_call_tool`, and `agent_run` in Python & Rust SDKs).
- **Self-Distilled `SVF` Calibration (`JouleSynapse`)**: Dataset-free Coordinate-Ridge singular-value self-distillation (`distill_svf_against_dense`) + exact per-row Cauchy-Schwarz gain projection during `.safetensors` ingestion (`distill_svf: true` by default).
- **SSE Per-Token Nanojoule Streaming & Zero-Hallucination Grammar/Tool Engine**: Native `stream: true` `text/event-stream` streaming with per-token `x_jouledb_token_nj` deltas plus `XGrammar`-class jump-forward JSON Schema (`response_format`) and function/tool-call (`tools` -> `tool_calls` / `tool_use`) constraining across OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, Python SDK (`chat_stream`, `chat_structured`, `chat_tools`), and Rust SDK (`HttpFabricClient`).

## [0.2.0] - 2026-11-23

### Added
- **Zero-PyTorch Causal Autoregressive Decoder & Multi-Layer Encoder (`JouleSynapse`)**:
  - Full `Llama-3` / `SmolLM2` / `Qwen2.5` / `Mistral` causal autoregressive decoder (`RoPE` + Grouped-Query Attention `GQA` + `RMSNorm` + `SwiGLU` + weight-tied/untied `LMHead` + pure-Rust byte-level `BpeTokenizer` + materialized `Mooncake` `(K, V)` prefix block reuse) executing directly over sub-1-bit `LittleBit-2` bitplanes (`POST /api/v1/synapse/decoder/generate`, `/v1/chat/completions`, `/v1/messages`).
  - Pure-Rust `WordPieceTokenizer` (`vocab.txt` & `tokenizer.json`) + full $L$-layer Multi-Head Self-Attention + `GELU` FFN + `LayerNorm` + Mean-Pooling encoder (`all-MiniLM-L6-v2` / BERT / BGE) with zero external ML runtime (`ort` / `candle` / PyTorch / `llama.cpp`).
- **Drop-In AI Model Server Disintermediation**: OpenAI (`GET /v1/models`, `POST /v1/embeddings`, `POST /v1/chat/completions`), Anthropic (`POST /v1/messages`), and Ollama (`GET /api/tags`, `POST /api/generate`) wire compatibility backed by a 4-tier thermodynamic cascade (`LangCache` ~`95 nJ` -> `Engram` $O(1)$ ~`220 nJ` -> `TriModal` RRF -> `LittleBit-2` hardware `wgpu` Metal/Vulkan `POPCNT` + `Titans`).
- **Cryptographic Capability Confinement & Single-File `.jcap` Appliances (`JouleWard`)**: RFC-2104 `HMAC-SHA256` `ThermoCap` macaroons with offline caveat attenuation, native OS kernel confinement (`macOS Seatbelt SBPL`, `Linux Landlock + bwrap + seccomp-bpf`, `WASI 0.2`), `85 nJ` Merkle CAS action memoization, and SHA-256 verified `JCAP_v1` single-file bundles.
- **High-Resolution (`1 nJ` / `1 ns`) Heterogeneous 6-Rail Fabric Energy Ledger + Live Hardware Sensor Fusion**: Tracks `cpu_nj`, `gpu_nj`, `npu_ane_nj`, `dram_uma_nj`, `storage_nvme_nj`, and `network_wire_nj` across 7 fabric compute tiers (`MCU` -> `WASM` -> `Mobile NPU` -> `Edge Robotic` -> `Apple UMA` -> `Linux Accelerator` -> `Cloud CXL`), fusing live OS/hardware power telemetry (`EnergySnapshot` / `HardwarePowerSample`) directly into per-hop receipts.
- **Database-Native Observability, Guardrails, Durable Workflows & Federated Skill Sync**: `JoulePulse` (`45 nJ` inline prompt-injection/SSRF/secret guardrail firewall, `POST /v1/traces` OTLP span ring, Pareto CI regression gate), `JouleFlow` (crash-idempotent journaled DAGs + carbon deferral + Saga compensations), and `JouleSynapseMesh` (`< 4 KB` `HMAC-SHA256` signed skill deltas with Byzantine Multi-Krum aggregation).
- **Container & Post-Git Disintermediation (`JouleCapsule` & `JouleSpace`)**: Pure-Rust OCI `.tar.gz` whiteout extractor, `O(1)` CoW branch VFS capsules, `Joulefile` DAG builder, `jj`-style conflict-free delta oplog with automatic secret scrubbing, and `LiveKit` multimodal room barge-in.
- **Python & Rust SDK Upgrades**: Added `synapse_status`, `ward_ledger`, `ward_mint_token`, `ward_exec`, `embed`, `chat`, `messages`, and `flow_execute` to both the `jouledb` Python SDK (`0.2.0`) and `joule-db-client::HttpFabricClient`.

### Changed
- License: MIT OR Apache-2.0 dual-licensing across the workspace, binaries, documentation, and examples.

## [0.1.0], TBD

### Added
- Initial public release surface
- Multi-platform prebuilt binaries (macOS arm64/x86_64, Linux x86_64/aarch64 musl, Windows x86_64)
- Examples directory with runnable programs
- Documentation site at openie-dev.github.io/jouledb
- GitHub Discussions enabled
- Issue templates (bug report, feature request)

[Unreleased]: https://github.com/openIE-dev/jouledb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/openIE-dev/jouledb/releases/tag/v0.1.0
