# JouleDB

**The thermodynamic nervous system and living-weights (`.jsyn`) database for AI, Embodied AI, and Autonomous Agents.**  
*Every query, branch fork, sensor join, container capsule, Kubernetes reconciliation, and sub-1-bit neural forward pass is metered in verifiable nanojoules (`nJ`).*

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Fleet](https://img.shields.io/badge/Hugging%20Face-82%20.jsyn%20Models-00E5FF)](https://huggingface.co/dcharlot65-openie)
[![Website](https://img.shields.io/badge/Web-jouledb.org-0A0A0F)](https://jouledb.org)

---

## Why JouleDB Exists

Modern AI and data stacks split state and intelligence across a dozen disconnected silos: an OLTP server, a warehouse, a vector index, a graph engine, a streaming cluster, a `dockerd` / `etcd` Kubernetes control plane, and a separate Python/CUDA server streaming `20–60 GB` frozen `.safetensors` or `.gguf` weight slabs across DRAM on every token.

**JouleDB collapses the stack into one safe-Rust substrate (`Apache-2.0 OR MIT`) built on two first principles:**

1. **The Verifiable Joule Is the Unit of Account:** Moving a byte across PCIe (`85 pJ/B`) or WAN (`2,400 pJ/B`) costs far more than on-package Unified Memory (`4 pJ/B`) or bitwise `POPCNT` arithmetic. JouleDB plans work across a deterministic complexity cascade (`LSN LangCache 95 nJ -> Engram O(1) 220 nJ -> TriModal BM25 ⋈ RaBitQ ⋈ HDC -> .jsyn Living Bitplanes`) and anchors 6-rail hardware energy receipts (`CPU`, `GPU`, `NPU/ANE`, `DRAM/UMA`, `NVMe`, `Wire`) in a Merkle ledger.
2. **Models Are Living SQL Tables (`.jsyn`), Not Frozen Encyclopedias:** Standalone LLMs waste ~85% of their dense FFN parameters acting as a lossy key-value database memorizing world facts. Inside JouleDB, **facts live in the database** (MVCC SQL + `Engram` + `TriModal GraphRAG` + `PathHD`), while **neural circuits live in `.jsyn` (`JSYN-v1`) sub-1-bit bitplanes**—shrinking `< 30 GB` open-weight checkpoints by **`20x–560x`** (`0.02–0.50 BPW`) with zero PyTorch, `llama.cpp`, or ONNX Runtime dependencies.

---

## The `.jsyn` (`JSYN-v1`) Living-Weights Ecosystem ([82 Public Models on Hugging Face](https://huggingface.co/dcharlot65-openie))

Standard scalar quantizers (`Q4_K_M`, `NVFP4`, `BitNet b1.58`) quantize entry-by-entry, hitting a hard `1.0 bit/weight` (`16x`) floor. `.jsyn` (`crates/joule-db-weights`) decomposes 2D and folded 3D Mixture-of-Experts projections via **`LittleBit-2 ⊕ Sakana SVF ⊕ Sparse Outliers`**:

$$W \approx \operatorname{diag}(s_{\text{row}}) \cdot \left( \sum_{k=1}^{r} \sigma_k z_k \, u_k v_k^\top \right) \cdot \operatorname{diag}(s_{\text{col}}) + S_{\text{outlier}}$$

- **Zero-Decompression Forward Pass:** Evaluates $y = \hat{W}x$ directly over packed `u64` FastLanes sign bitplanes using L1-resident **`T-MAC` byte lookup tables** (CPU) or `wgpu` **`countOneBits` (`XNOR-POPCNT`)** compute shaders (Metal/Vulkan/WebGPU) in $O((m + n) \cdot r)$—and evaluates only active top-$k$ MoE slices in $O(k)$.
- **Exact Structural Fidelity + Live Plasticity:** Keeps 1D `RMSNorm`, `+1` Gemma offsets, `QK-Norm`, `Liquid ShortConv` `1x3` kernels, and `Mamba-2` SSM vectors in **exact `FP32`**, embeds BPE/SentencePiece tokenizers, calibrates channel skew via **`AWQ ⊕ GPTQ` Empirical-Bayes `SVF`** (`92.04%` MSE reduction at `0` extra bits), and supports **Google `Titans`** test-time plasticity with MVCC `rollback_to_lsn`.
- **82 Permissive `< 30 GB` Models Ready on Hugging Face (`dcharlot65-openie`):** Spanning **Google `Gemma 4`** (`E2B`, `E4B`, `26B-A4B`, `EmbeddingGemma-2`), **Thinking Machines Lab** (`Inkling-2-Mini` & `TML-d24-VarLoop` `50.3 GB -> 89.8 MB = 560.5x`), **NVIDIA** (`Nemotron-3-Nano 4B/30B`, `Dory 1.0`, `Sigma-3B`, `Qwen3.5-Draft`, `Eagle3`, `Cosmos-Embed1`, `Parakeet-TDT`), **Poolside** (`Laguna-XS-2.1` `33.4B` MoE `63.8 GB -> 176.3 MB = 361.8x`), **Liquid AI** (`LFM3`, `LFM3-Omni`, `d1-2B`), **PrismML** (`Ternary-Bonsai-2`), **Cactus Compute**, **Microsoft `BitNet` & `Aion`**, **AllenAI `OLMo-2`**, **IBM `Granite 4.1`**, **Sakana AI**, and **HuggingFace `SmolVLA` / `SmolVLM2`**.
- Full binary specification & rank-sweep theory: **[`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)**.

### Live Apple M5 Max Rank-Sweep Proof (`POST /api/v1/synapse/bench/rank_sweep`)

| Checkpoint & Projection | Rank `r` | `BPW` | Shrinkage vs `FP16` | Latency (`B=32`) | UMA Energy / Sample | Energy Savings | Output Cosine / `KL` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Gemma-4-E2B-IT`** `q_proj` (`2048x1536` `BF16`) | `8` | `0.0284` | **`564.2x`** | `384.6 µs` | `49.8 nJ` | **`562.3x`** | `cos 0.271` / `KL 3.87` |
| **`Gemma-4-E2B-IT`** `q_proj` (`2048x1536` `BF16`) | `32` | `0.0558` | **`286.6x`** | `830.9 µs` | `108.5 nJ` | **`258.1x`** | `cos 0.341` / `KL 3.69` |
| **`Gemma-4-E2B-IT`** `q_proj` (`2048x1536` `BF16`) | `64` | `0.0924` | **`173.1x`** | `1,722.9 µs` | `186.7 nJ` | **`149.9x`** | `cos 0.392` / `KL 3.37` |
| **`Gemma-4-E2B-IT`** `q_proj` (`2048x1536` `BF16`) | `128` | `0.1657` | **`96.6x`** | `3,193.2 µs` | `343.2 nJ` | **`81.6x`** | `cos 0.455` / `KL 2.72` |
| **`Inkling-2-Mini`** `w13` (`128`-Exp `NVFP4` MoE) | `8 -> 128` | `0.023 -> 0.134` | **`701.3x -> 119.5x`** | `466 -> 4,156 µs` | `60.1 -> 416.2 nJ` | **`698.6x -> 100.9x`** | `cos 0.113 -> 0.350` |
| **`SmolLM2-135M`** (Full 30-Layer Causal LLM) | `8 -> 128` | `0.125 -> 0.388` | **`65.2x -> 27.4x`** | **`58.4 -> 31.3 tok/s`** | `3.93 -> 9.35 MB .jsyn` | Lossless draft verified | `w_cos 0.192 -> 0.622` |

---

## Core Database, Embodied & Cloud-Native Capabilities (`v0.2.9`)

- **7 Query Languages + Out-of-Core OLAP:** SQL, Cypher, CQL, GraphQL, Datalog, SPARQL, Gremlin (+ InfluxQL/PromQL + `SigQL` biosignals), `2,048`-row **Morsel Grace Out-of-Core Spill-to-Disk** joins/aggregates, **`ClickBench`/`Vortex` Granule `ZoneMap + SelectionVector`** (`97.96%` cold skip in `164 µs`), bit-exact `ALP + FSST` codecs, `Arrow Flight SQL` (`JDB_IPC1`), and `DuckLake` + `Iceberg v3` `AS OF` time-travel.
- **Formally Verified ACID/SSI, Financial Ledger & Reactive DAGs:** Cahill `SSI` certified by **Kleppmann `Hermitage` (`6/6` clean in `1.13 ms`)** and **Jepsen/`Elle` Tarjan `SCC` cycle detector (`0` cycles)**, **`sqllogictest` MD5 + `SQLancer` (`TLP` & `NoREC`)** (`0` violations), **`TigerBeetle`-class `i128` linked two-phase accounting + `VOPR` torn-WAL/bit-rot simulator**, and **Goldman Sachs `SecDB` + Google `Napa`** dirty-cone freshness-SLA DAGs (`99.23%` compute saved).
- **360° Hybrid GraphRAG & Neurosymbolic HDC:** `TriModal RRF` (`BM25 ⋈ RaBitQ 1-bit POPCNT ⋈ 15-Primitive HDC` at `32x` less vector RAM) + `ACORN` 2-hop filtered rescue + `HippoRAG` Personalized PageRank + `PathHD` (**`70.0%` `WN18RR` `Hits@1` at `10 µJ/pred`**, **`100%` `LongMemEval-60` at `~0.8 µJ/turn`**).
- **Physical AI, Robotics & HMI:** `EctStore` (`HLC` + `LSL` sync), `kdb+`-class strictly causal `ASOF` joins (`0.63 µs/query`, `0` future leakage), `SE(3)` quaternion `SLERP` kinematic trees, active-inference `SurpriseGate`, and Feldera-class `DBSP` `ZSet<T>` $O(|\Delta|)$ incremental views.
- **Post-Docker `JouleCapsule`, Post-Git `JouleSpace`, `JouleWard` & `JouleKube`:** Daemonless pure-Rust OCI v2 CAS + $O(1)$ CoW container+DB forks (`1.13 µs/branch`), `Jujutsu`/`Zed Delta` op-logs, RFC-2104 `ThermoCap` capability macaroons, and `kubectl`-native Kubernetes + CNCF AI-Conformance (`DRA`, `Kueue`, `LWS`, `InferencePool`, `agent-sandbox`, `Kagent`) backed by MVCC SQL (`0` `etcd`, `0.0W` warm pools).
- **Wire Surfaces:** Native `JWP`, PostgreSQL `pgwire`, Arrow Flight SQL, MCP 2026 Streamable HTTP, WebSocket, REST, OpenAI `/v1/*`, Anthropic `/v1/messages`, Ollama `/api/*`, and Kubernetes `/api/v1` + `/apis/*`.

---

## Quickstart

```bash
# 1. Install prebuilt binary via cargo-binstall (or pull ghcr.io/openie-dev/jouledb:0.2.9)
cargo binstall jouledb

# 2. Launch JouleDB server (HTTP :8080, TCP :8081, PgWire :8082, JWP :8083)
jouledb --port 8080 --data ./joule-data

# 3. Pull a sub-1-bit .jsyn model from OpenIE's 82-model Hugging Face Hub & query its bitplanes in SQL
curl -s http://127.0.0.1:8080/api/v1/synapse/hf/pull \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"dcharlot65-openie/TML-d24-VarLoop-LittleBit","model_id":"tml-varloop"}'

jouledb query "SELECT tensor_name, rows, cols, bits_per_weight FROM synapse_tensors LIMIT 5"

# 4. Run the 7-pillar 360° database & formal isolation certification suite
curl -s -X POST http://127.0.0.1:8080/api/v1/bench/industry_suite \
  -H "Content-Type: application/json" -d '{}'
```

### Build & Verify from Source

```bash
cargo build --release -p joule-db-server --bin jouledb
cargo test  -p joule-db-core --lib
cargo test  -p joule-db-server --test wave24_tigerbeetle_secdb_napa_zonemap_industry_suite_e2e
cargo test  -p joule-db-server --test wave25_rank_sweep_pareto_and_svf_calibration_e2e
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
- **OpenIE Hugging Face `.jsyn` Hub (`82` Models, Spaces & Benchmark Dataset):** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Whitepaper (`v0.3`, November 2026):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Living-Weights Binary Spec:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **360° Global Ecosystem SWOT:** [`docs/jouledb/SOTA-SWOT-2026-11.md`](./docs/jouledb/SOTA-SWOT-2026-11.md)
- **SDKs, Query Languages & Runbooks:** [`docs/jouledb/`](./docs/jouledb/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
