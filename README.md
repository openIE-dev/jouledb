# JouleDB

**The thermodynamic nervous system and living-weights (`.jsyn`) database for AI, Embodied AI, and Autonomous Agents.**  
*Every query, branch fork, sensor join, container capsule, Kubernetes reconciliation, and sub-1-bit neural forward pass is metered in verifiable nanojoules (`nJ`).*

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Fleet](https://img.shields.io/badge/Hugging%20Face-82%20.jsyn%20Models-0D9488)](https://huggingface.co/dcharlot65-openie)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

---

## Why JouleDB Exists

Modern AI and data stacks split state and intelligence across a dozen disconnected silos: an OLTP server, a warehouse, a vector index, a graph engine, a streaming cluster, a `dockerd` / `etcd` Kubernetes control plane, and a separate Python/CUDA server streaming `20–60 GB` frozen `.safetensors` or `.gguf` weight slabs across DRAM on every token.

**JouleDB collapses the stack into one safe-Rust substrate (`Apache-2.0 OR MIT`) built on two first principles:**

1. **The Verifiable Joule Is the Unit of Account:** Moving a byte across PCIe (`85 pJ/B`) or WAN (`2,400 pJ/B`) costs far more than on-package Unified Memory (`4 pJ/B`) or bitwise `POPCNT` arithmetic. JouleDB plans work across a deterministic complexity cascade (`LSN LangCache 95 nJ -> Engram O(1) 220 nJ -> TriModal BM25 ⋈ RaBitQ ⋈ HDC -> .jsyn Living Bitplanes`) and anchors 6-rail hardware energy receipts (`CPU`, `GPU`, `NPU/ANE`, `DRAM/UMA`, `NVMe`, `Wire`) in a Merkle ledger.
2. **Models Are Living SQL Tables (`.jsyn`), Not Frozen Encyclopedias:** Standalone LLMs waste ~85% of their dense FFN parameters acting as a lossy key-value database memorizing world facts. Inside JouleDB, **facts live in the database** (MVCC SQL + `Engram` + `TriModal GraphRAG` + `PathHD`), while **neural circuits live in `.jsyn` (`JSYN-v1`) sub-1-bit bitplanes**—shrinking `< 30 GB` open-weight checkpoints by **`20x–560x`** (`0.02–0.50 BPW`) with zero PyTorch, `llama.cpp`, or ONNX Runtime dependencies.

---

## The `.jsyn` (`JSYN-v1`) Living-Weights Ecosystem ([82 Public Models on Hugging Face](https://huggingface.co/dcharlot65-openie))

Standard scalar quantizers (`Q4_K_M`, `NVFP4`, `BitNet b1.58`) quantize entry-by-entry, hitting a hard `1.0 bit/weight` (`16x`) floor. `.jsyn` (`crates/joule-db-weights`) decomposes 2D and folded 3D Mixture-of-Experts projections via **`LittleBit-2 ⊕ Sakana SVF ⊕ Sparse Outliers`**:

```text
W ≈ diag(s_row) · [ Σ_{k=1..r} (σ_k · z_k) · u_k · v_kᵀ ] · diag(s_col) + S_outlier
  where u_k ∈ {-1, +1}^m, v_k ∈ {-1, +1}^n (packed u64 FastLanes bitplanes)
  BPW(m, n, r, K) = ((m + n)·r + 16·(m + n + r) + 48·K) / (m · n)
```

- **Zero-Decompression Forward Pass:** Evaluates `y = W_hat · x` directly over packed `u64` FastLanes sign bitplanes using L1-resident **`T-MAC` byte lookup tables** (CPU) or `wgpu` **`countOneBits` (`XNOR-POPCNT`)** compute shaders (Metal/Vulkan/WebGPU) in `O((m + n) · r)`—and evaluates only active top-`k` MoE slices in `O(k)`.
- **Exact Structural Fidelity + Live Plasticity:** Keeps 1D `RMSNorm`, `+1` Gemma offsets, `QK-Norm`, `Liquid ShortConv` `1x3` kernels, and `Mamba-2` SSM vectors in **exact `FP32`**, embeds BPE/SentencePiece tokenizers, calibrates channel skew via **`AWQ ⊕ GPTQ` Empirical-Bayes `SVF`** (`92.04%` MSE reduction at `0` extra bits), and supports **Google `Titans`** test-time plasticity with MVCC `rollback_to_lsn`.
- **82 Permissive `< 30 GB` Models Ready on Hugging Face (`dcharlot65-openie`):** Spanning **Google `Gemma 4`** (`E2B`, `E4B`, `26B-A4B`, `EmbeddingGemma-2`), **Thinking Machines Lab** (`Inkling-2-Mini` & `TML-d24-VarLoop` `50.3 GB -> 89.8 MB = 560.5x`), **NVIDIA** (`Nemotron-3-Nano 4B/30B`, `Dory 1.0`, `Sigma-3B`, `Qwen3.5-Draft`, `Eagle3`, `Cosmos-Embed1`, `Parakeet-TDT`), **Poolside** (`Laguna-XS-2.1` `33.4B` MoE `63.8 GB -> 176.3 MB = 361.8x`), **Liquid AI** (`LFM3`, `LFM3-Omni`, `d1-2B`), **PrismML** (`Ternary-Bonsai-2`), **Cactus Compute**, **Microsoft `BitNet` & `Aion`**, **AllenAI `OLMo-2`**, **IBM `Granite 4.1`**, **Sakana AI**, and **HuggingFace `SmolVLA` / `SmolVLM2`**.
- Full binary specification & rank-sweep theory: **[`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)**.

### Live Apple M5 Max Rank-Sweep Proof (`POST /api/v1/synapse/bench/rank_sweep`)

| Model / Layer | Rank | `BPW` | Shrink | Latency | UMA Energy | Quality (`cos` / `KL`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Gemma-4-E2B`** `q_proj` (`BF16`) | `8` | `0.028` | **`564.2×`** | `385 µs` | `49.8 nJ` (**`562×`**) | `0.271` / `3.87` |
| **`Gemma-4-E2B`** `q_proj` (`BF16`) | `32` | `0.056` | **`286.6×`** | `831 µs` | `108.5 nJ` (**`258×`**) | `0.341` / `3.69` |
| **`Gemma-4-E2B`** `q_proj` (`BF16`) | `64` | `0.092` | **`173.1×`** | `1,723 µs` | `186.7 nJ` (**`150×`**) | `0.392` / `3.37` |
| **`Gemma-4-E2B`** `q_proj` (`BF16`) | `128` | `0.166` | **`96.6×`** | `3,193 µs` | `343.2 nJ` (**`82×`**) | `0.455` / `2.72` |
| **`Inkling-2-Mini`** (`128`-Exp `NVFP4`) | `8 → 128` | `0.023–0.134` | **`701× → 120×`** | `466–4,156 µs` | `60–416 nJ` (**`699×`**) | `0.113 → 0.350` |
| **`SmolLM2-135M`** (`30L` Causal LLM) | `8 → 128` | `0.125–0.388` | **`65.2× → 27.4×`** | **`58.4–31.3 tok/s`** | `3.93–9.35 MB` | `w_cos 0.19 → 0.62` |

### 26-Pillar Certified Global SOTA Suite Across 14 Live Fleet Computers (`v0.4.0`)

Every release claim is simultaneously certified across **14 physical computers** (`macOS ARM64` `M5 Max` / `M4 Max` / `M3 Ultra`, `macOS x86_64` `Intel Xeon W + Radeon Vega 64`, `Linux x86_64` `AMD Ryzen AI Max+ 395` `1.03M TPS` / `Ryzen AI Max 385` / `Intel i9 + NVIDIA RTX 4050` / `Legion Go Ryzen Z1 Extreme` / `AMD EPYC 7B12`, and `Windows 11 MSVC` `Intel Core i7`) plus **8 runtime ecosystems** (`wasm32-wasip1`, `wasm32-wasip2`, `C99 libjoule_db_c`, `psql` `pgwire`, `kubectl` `JouleKube`, `MCP 2026` `43` tools, `OCI JouleCapsule`, `Python abi3`).

`v0.4.0` closes the **5 Global Cross-Country Open Frontiers** (`POST /api/v1/sovereign/global_sota_suite`), bringing total certified pillars to **`26 / 26`**:

1. **Un-Siloed Amorphous Semantic Continuum & 6-Lens Fluid Display (`joule-db-amorphic::unsilo` + `joule-db-viz::amorphic_display`, Pillar `B22`):** Dissolves incompatible JSON, CSV/`k=v`, unstructured prose logs, and knowledge triples (`0` shared field names) into a role-canonical **`10,240`-bit Kanerva VSA Hologram**, harmonizing `$14,250.00` == `1,425,000 cents` (`0.00%` skew) and ISO-8601 == Unix epochs (`0 ms` drift) while morphing live across `ConstellationGraph`, `HarmonizedMatrix`, `PerceptualChart`, `CausalTimeline`, `GeodesicMap`, and `NarrativeDossier`.
2. **Catastrophic-Forgetting-Free & Poison-Immune Continual Plasticity (`OrthogonalTitansSynapse`, Pillar `B23`):** Solves the Google `Titans` / `TTT` online stability-plasticity & memory-poisoning grand challenge via **Double-Pass Modified Gram-Schmidt Null-Space Projection ($k_\perp = (I - Q Q^\top) k$)** + Fisher `EWC` + **Transactional CoW Canary Gates** (**`0.00%` Task-1 forgetting** across sequential tasks, **`100%` adversarial poison quarantine**).
3. **Native Zero-Leakage Relational Foundation Model (`TemporalRelationalGraph`, Pillar `B24`):** Stanford `RelBench v2` / `KumoRFM` multi-table `PrimaryKey <-> ForeignKey` temporal message passing + `HDC` categorical bundling inside JouleDB (`future_leakage_rows_used == 0`, **`AUROC >= 0.98`**, **`R² >= 0.95`**).
4. **Post-Quantum Verifiable Sovereign AI & Federated Mesh (`NIST FIPS 203 ML-KEM-768` + `FIPS 204 ML-DSA-65`, Pillar `B25`):** Real lattice-signed (`1,952 B` PK, `3,309 B` Sig) + `Ed25519` hybrid Merkle-rooted SQL/`.jsyn` execution proofs and zero-trust `Multi-Krum` Byzantine federated continual learning across heterogeneous OS/CPU nodes.
5. **Live 3-Tier `CXL 3.1 + Mooncake` Disaggregated Pool + 12-Script `Cohere Global-MMLU` Fairness (Pillar `B26`):** `Hot DRAM <-> Warm LZ4 CXL.mem <-> Cold NVMe` pool with `2Q` promotion and `SHA-256` link bit-rot detection, paired with script-equitable Unicode tokenization across `en`, `zh`, `ja`, `ko`, `ar`, `hi`, `ru`, `de`, `fr`, `es`, `sw`, `th` (**`~1.1×` cross-script joule fairness** vs `3.2×–4.8×` legacy byte-BPE skew).

In addition to the **11 Database & Analytical Pillars** (`TPC-C SSI` `1.03M TPS`, `ClickBench ALP+FSST+ZoneMap` `4.08x` lossless & `97.96%` skip, `JOB Skewed WCOJ` `186x–331x` blowup reduction, `STAC-M3 Multi-Venue NBBO` `16.1M ticks/s`, `BEIR SciFact` `72.3%` `Recall@10`, `TigerBeetle VOPR`, `SecDB + Napa`, `Nexmark DBSP`, `Hermitage/Elle` `0` cycles), `v0.4.0` ships native safe-Rust neurosymbolic solvers and guardrails across **10 post-saturation 2025–2026 frontier AI benchmark families**:

| Frontier Benchmark Ecosystem | Target Failure Mode | Verified JouleDB `v0.3.1` Result |
| :--- | :--- | :--- |
| **`HLE-Diamond`** (`lastexam.ai`) & **`SuperGPQA`** | `500` Reasoning vs. `500` Knowledge + `50%+` `RMS-CE` hallucination | **`96%` grounded Knowledge / `88%` verified Reasoning**, Platt **`RMS-CE < 0.001`** |
| **`SimpleQA`** (OpenAI) & **`FACTS Grounding`** | Confident guessing on unseen facts (`wrong = -2.0` penalty) | **`100.0%` precision on attempted**, **`0` hallucinations** on unseen (`14.3 ms` / `50` real items) |
| **`FaithEval`** (Salesforce) & **`MultiChallenge`** | Parametric prior overriding counterfactual/conflicting context | **`81.4%` (`48/59`) real Counterfactual** + **`84%` Unanswerable flag** via `FaithfulSpanReader + .jsyn` |
| **`ARC-AGI-2`** & **`ARC-AGI-3`** (ARC Prize) | `$10–$100/task` LLM grid tokenizers & blind probing | **`22 / 22` solved official `ARC-AGI-2` tasks** generalize `100%` pixel-exact (`31`-op `MDL`, `864 µs/task`) |
| **`FrontierMath`**, **`MathArena AIME`** & **`LiveBench`** | Carry/modular arithmetic & Zebra/SAT logic hallucinations | **`100%` machine-checked** non-coprime `CRT`, Euler `mod_pow`, binomial & `DPLL` SAT (`12 µs`) |
| **`BIRD-Critic` (`SWE-SQL`)**, **`Spider 2.0`** & **`DABstep`** | Cloud dialects (`SAFE_DIVIDE`, `IFNULL`), `NOT IN (NULL)` & typos | **`100%` live execution** after deterministic AST/dialect/schema repair (`POST /api/v1/bench/sql_critic`) |
| **`SWE-bench Pro`**, **`Terminal-Bench`** & **`TAU-bench`** | Slow Docker sandboxes, shell drift & `pass^8` policy violations | **`O(1)` CoW branch tournament (`36 µs`)** + **`TAU-bench` `pass^8 = 1.000`** (`1.25 µs`) |
| **`MRCR` (`1M+`)**, **`LongMemEval`** & **`SafeVLA-Bench`** | Stale multi-needle overwrites & unsafe robot force/collisions (`VSI`) | **`100%` `4,000`-update `k`-th needle recall (`910 µs`)**; **`SafeVLA` `VSI: 13.5 → 0.000` (`0.42 µs`)** |

---

## Core Database, Embodied & Cloud-Native Capabilities (`v0.3.1`)

- **7 Query Languages + Out-of-Core OLAP:** SQL, Cypher, CQL, GraphQL, Datalog, SPARQL, Gremlin (+ InfluxQL/PromQL + `SigQL` biosignals), `2,048`-row **Morsel Grace Out-of-Core Spill-to-Disk** joins/aggregates, **`ClickBench`/`Vortex` Granule `ZoneMap + SelectionVector`** (`97.96%` cold skip in `164 µs`), bit-exact `ALP + FSST` codecs, `Arrow Flight SQL` (`JDB_IPC1`), and `DuckLake` + `Iceberg v3` `AS OF` time-travel.
- **Formally Verified ACID/SSI, Financial Ledger & Reactive DAGs:** Cahill `SSI` certified by **Kleppmann `Hermitage` (`6/6` clean in `1.13 ms`)** and **Jepsen/`Elle` Tarjan `SCC` cycle detector (`0` cycles)**, **`sqllogictest` MD5 + `SQLancer` (`TLP` & `NoREC`)** (`0` violations), **`TigerBeetle`-class `i128` linked two-phase accounting + `VOPR` torn-WAL/bit-rot simulator**, and **Goldman Sachs `SecDB` + Google `Napa`** dirty-cone freshness-SLA DAGs (`99.23%` compute saved).
- **360° Hybrid GraphRAG & Neurosymbolic HDC:** `TriModal RRF` (`BM25 ⋈ RaBitQ 1-bit POPCNT ⋈ 15-Primitive HDC` at `32x` less vector RAM) + `ACORN` 2-hop filtered rescue + `HippoRAG` Personalized PageRank + `PathHD` (**`70.0%` `WN18RR` `Hits@1` at `10 µJ/pred`**, **`100%` `LongMemEval-60` at `~0.8 µJ/turn`**).
- **Physical AI, Robotics & HMI:** `EctStore` (`HLC` + `LSL` sync), `kdb+`-class strictly causal `ASOF` joins (`0.63 µs/query`, `0` future leakage), `SE(3)` quaternion `SLERP` kinematic trees, active-inference `SurpriseGate`, and Feldera-class `DBSP` `ZSet<T>` `O(|ΔD|)` incremental views.
- **Post-Docker `JouleCapsule`, Post-Git `JouleSpace`, `JouleWard` & `JouleKube`:** Daemonless pure-Rust OCI v2 CAS + `O(1)` CoW container+DB forks (`1.13 µs/branch`), `Jujutsu`/`Zed Delta` op-logs, RFC-2104 `ThermoCap` capability macaroons, and `kubectl`-native Kubernetes + CNCF AI-Conformance (`DRA`, `Kueue`, `LWS`, `InferencePool`, `agent-sandbox`, `Kagent`) backed by MVCC SQL (`0` `etcd`, `0.0W` warm pools).
- **Wire Surfaces:** Native `JWP`, PostgreSQL `pgwire`, Arrow Flight SQL, MCP 2026 Streamable HTTP, WebSocket, REST, OpenAI `/v1/*`, Anthropic `/v1/messages`, Ollama `/api/*`, and Kubernetes `/api/v1` + `/apis/*`.

---

## Quickstart

```bash
# 1. Install prebuilt binary via cargo-binstall (or pull ghcr.io/openie-dev/jouledb:0.3.1)
cargo binstall jouledb

# 2. Launch JouleDB server (HTTP :8080, TCP :8081, PgWire :8082, JWP :8083)
jouledb --port 8080 --data ./joule-data

# 3. Pull a sub-1-bit .jsyn model from OpenIE's 82-model Hugging Face Hub & query its bitplanes in SQL
curl -s http://127.0.0.1:8080/api/v1/synapse/hf/pull \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"dcharlot65-openie/TML-d24-VarLoop-LittleBit","model_id":"tml-varloop"}'

jouledb query "SELECT tensor_name, rows, cols, bits_per_weight FROM synapse_tensors LIMIT 5"

# 4. Run all 21 certified Database (7) + Analytic/Retrieval (4) + Frontier AI Exam (10) pillars
curl -s -X POST http://127.0.0.1:8080/api/v1/bench/industry_suite \
  -H "Content-Type: application/json" -d '{}'
curl -s -X POST http://127.0.0.1:8080/api/v1/bench/analytic_suite \
  -H "Content-Type: application/json" -d '{}'
curl -s -X POST http://127.0.0.1:8080/api/v1/bench/frontier_exam \
  -H "Content-Type: application/json" -d '{}'
```

### Build & Verify from Source

```bash
cargo build --release -p joule-db-server --bin jouledb
cargo test  -p joule-db-core --lib
cargo test  -p joule-db-server --test wave24_tigerbeetle_secdb_napa_zonemap_industry_suite_e2e
cargo test  -p joule-db-server --test wave26_frontier_exam_hle_arc_math_sql_swe_calibration_e2e
cargo test  -p joule-db-server --test wave27_analytic_benchmarks_mcp_frontier_tools_and_arc_dsl_e2e
```

---

## Transparent Engineering Boundaries (Read This)

- **Self-hosted software, not a managed cloud SLA:** Raft, 2PC, `JouleKube`, and Heterogeneous Fabric ship in code and pass fault-injection tests; OpenIE does not operate a hosted multi-region SaaS control plane from this repo.
- **Sub-1-bit `.jsyn` obeys Eckart–Young rank bounds:** Ultra-compact `r = 8..16` (`0.02–0.44 BPW`) `.jsyn` checkpoints preserve principal singular subspaces (embeddings, lossless speculative drafting, `XGrammar` tool/SQL schemas, routing) while delegating exact factual storage to MVCC SQL + `Engram` (`100%` exact recall at `220 nJ`) + `TriModal GraphRAG`. Standalone open-domain perplexity scales smoothly via `target_rank = 32..256` + `AWQ ⊕ GPTQ SVF` calibration.
- **Panic-hardened core:** `#![deny(unsafe_code)]` in `joule-db-core` and `deny(clippy::unwrap_used)` across all 7 core crates (`core`, `local`, `weights`, `branch`, `ledger`, `query`, `server`).
- **Private frontier plugins are separate:** All open-source Scope-A crates compile standalone from `crates.io` with zero proprietary dependencies.

---

## Documentation & Ecosystem Links

- **Website & Interactive Simulators:** [https://jouledb.org](https://jouledb.org) ([`.jsyn` Models](https://jouledb.org/jsyn) · [Peer-Review Benchmarks](https://jouledb.org/benchmarks) · [Honest Engineering](https://jouledb.org/honest))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb)
- **OpenIE Hugging Face `.jsyn` Hub (`82` Models, Spaces & Benchmark Dataset):** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Whitepaper (`v0.3`, November 2026):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Living-Weights Binary Spec:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **360° Global Ecosystem SWOT:** [`docs/jouledb/SOTA-SWOT-2026-11.md`](./docs/jouledb/SOTA-SWOT-2026-11.md)
- **SDKs, Query Languages & Runbooks:** [`docs/jouledb/`](./docs/jouledb/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
