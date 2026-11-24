# JouleDB: The Thermodynamic Nervous System & Living-Weights Substrate for Intelligent Computing

## Whitepaper Specification — Version 0.3 (`v0.2.9`, November 2026)
**Open Interface Engineering, Inc. (`OpenIE`)**  
**Website:** [https://jouledb.org](https://jouledb.org) | **GitHub:** [openIE-dev/jouledb](https://github.com/openIE-dev/jouledb) | **Hugging Face Fleet (`82 .jsyn`):** [huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)  
**License:** `Apache-2.0 OR MIT`

> Supersedes [WHITEPAPER-JOULEDB-2026-05.md](WHITEPAPER-JOULEDB-2026-05.md) (`v0.2`, May 2026) and [WHITEPAPER-JOULEDB-2026-04.md](WHITEPAPER-JOULEDB-2026-04.md) (`v0.1`). Documents the complete late-2026 architecture shipped through `v0.2.9`: **`.jsyn` (`JSYN-v1`) Sub-1-Bit Living Model Weights** (`82` public models on Hugging Face), **`JouleKube`** (Kubernetes-native AI control plane with `0` `etcd`), **Post-Docker `JouleCapsule` & Post-Git `JouleSpace`**, **`JouleWard`** (`ThermoCap` capability macaroons & 6-rail hardware-fused `nJ` accounting), **Embodied `ECT` / causal `ASOF` / `SE(3)` / `SigQL`**, and **360° Formal Database Certification** (`Hermitage/Elle` `SSI`, `SQLancer TLP/NoREC`, `TigerBeetle VOPR`, Goldman `SecDB` + Google `Napa` reactive DAGs, Morsel Out-of-Core Grace Spill, and `ClickBench ZoneMaps`).

---

## Abstract

For fifty years, software infrastructure separated state and intelligence into disconnected silos: a relational OLTP database, an analytical warehouse, a vector index, a graph store, a streaming engine, a Docker/Kubernetes container daemon, and a multi-gigabyte frozen neural checkpoint server. Data movement across those boundaries burns orders of magnitude more physical energy and latency than arithmetic.

**JouleDB** collapses the entire stack into **one panic-hardened Rust substrate** (`#![deny(unsafe_code)]` in `joule-db-core`, `deny(clippy::unwrap_used)` across all 7 core crates) where **every byte, row, vector, hypervector, kinematic pose, container rootfs, Kubernetes manifest, and neural weight bitplane** shares a single Copy-on-Write (`CoW`) Serializable Snapshot Isolation (`SSI`) engine—and every operation produces a cryptographically verifiable 6-rail nanojoule (`nJ`) receipt.

---

## 1. Five Unified Organs of JouleDB (`v0.2.9`)

1. **ACID/SSI Multi-Model, Out-of-Core OLAP & Financial Ledger Core (`joule-db-core`, `joule-db-query`):**
   - Lock-free positional `pread`/`pwrite`, `IoRingEngine` extent coalescing, and full Cahill `SSI` (`rw`-antidependency detection) formally verified by **Kleppmann `Hermitage` (`6/6` anomalies clean in `1.13 ms`)** and **Jepsen/`Elle` Tarjan $O(V+E)$ `SCC` cycle analysis (`0` cycles)**.
   - **Morsel-Driven Out-of-Core Grace Hash Join & GroupBy** (`2,048`-row `ColumnMorsel` batches spilling binary CRC32 `JMSB` partitions under strict RAM caps), **`ClickBench`/`Vortex` Granule `ZoneMap` + `SelectionVector`** (`97.96%` cold granule skip in `164 µs`), bit-exact **`ALP` + `FSST`** codecs (`>6.4x`), **`Arrow Flight SQL` (`JDB_IPC1`)**, **`DuckLake` + `Iceberg v3`** `AS OF` snapshots, and **`sqllogictest` MD5 + `SQLancer` (`TLP` & `NoREC`)** differential logic-bug verification (`0` violations).
   - **`TigerBeetle`-Class Linked Two-Phase Double-Entry Accounting + `VOPR` Deterministic Fault Simulator** (`i128` accounts, `PendingReserve`/`PostPending`/`VoidPending` partial settlement, atomic `linked` rollback, torn-WAL + 1-bit NAND bit-rot conservation) and **Goldman Sachs `SecDB` Dirty-Cone DAG + Google `Napa` Freshness-SLA Controller** (`99.23%` compute saved via early cutoff + burst write coalescing).
2. **360° Retrieval, Neurosymbolic HDC & Zero-Tax Agent Memory (`joule-db-amorphic`, `joule-db-hdc`):**
   - **`TriModal RRF`** (`Okapi BM25 ⋈ RaBitQ 1-bit SIMD POPCNT ⋈ 15-Primitive HDC` at `~32x` smaller vector RAM than `FP32` HNSW) + **`ACORN` 2-hop filtered bridge rescue** + **`HippoRAG` Personalized PageRank** + auto-sharded unbounded `AmorphicStore` holograms.
   - **`PathHD` 3-Hop GHRR Link Prediction:** **`70.0%` `Hits@1` / `88.1%` `Hits@10` (`MRR = 0.7628`) on `WN18RR`** at **`10 µJ/prediction`** (`0` gradient descent), and **`100.0%` (`5/5`) on `LongMemEval-60`** bitemporal contradiction/abstention recall in **`277.4 µs/probe` (`~0.8 µJ/turn`)**.
   - **Declarative `LOTUS`-Class `SEM_*` SQL & Governed `SemanticCatalog`:** Thompson-sampled Pareto bandit cascade (`LSN LangCache -> Engram -> HDC/TriModal -> .jsyn`) + deterministic BFS foreign-key join & row-level tenant compiler (`0%` join hallucination).
3. **`.jsyn` (`JSYN-v1`) Living Model Weights & Native AI Harness (`joule-db-weights`, `joule-db-gpu`):**
   - Compiles `.safetensors` (`FP32`/`FP16`/`BF16`/`FP8`/`NVFP4`/MLX `U32`/3D MoE) and `.gguf` (`Q1_0_G128`, `I2_S`, `PQ2_0`, `TQ1_0`, `K`-quants) checkpoints into SQL-queryable sub-1-bit **`LittleBit-2 ⊕ Sakana SVF`** bitplanes (`0.02–0.50 BPW`, **`20x–560x` smaller than `FP16`**), published across **[82 public `.jsyn` repositories on Hugging Face](https://huggingface.co/dcharlot65-openie)**.
   - Executes causal LLMs (`Llama-3`, `Gemma 4`, `Qwen3/3.5`, `Thinking Machines Inkling-2 & TML-d24-VarLoop`, `NVIDIA Nemotron-H Mamba-2 / Dory / Sigma`, `Liquid AI LFM3 ShortConv`, `PrismML Ternary-Bonsai-2`, `Poolside Laguna-XS-2.1`, `BitNet b1.58`), bidirectional embedders (`MiniLM`, `ModernBERT`, `EmbeddingGemma-2`), vision (`SigLIP`), and speech (`Whisper` 400-FFT Seq2Seq) in **pure safe Rust + `wgpu` Metal/Vulkan/WebGPU `XNOR-POPCNT` shaders**—with **zero PyTorch, `llama.cpp`, or ONNX Runtime**.
   - Includes **Empirical-Bayes `AWQ ⊕ GPTQ` Activation-Aware `SVF` Calibration** (`92.04%` skewed-activation MSE reduction at `0` extra bits), **Google `Titans`** surprise-gated fast weights with MVCC `rollback_to_lsn`, **`Engram`** $O(1)$ factual memory, **`Mooncake`** Radix KV prefix caching, lossless self-speculative **`T-MAC`** decoding, and **`XGrammar`** zero-hallucination JSON/tool DFAs.
   - See [JSYN-LIVING-WEIGHTS-SPEC.md](JSYN-LIVING-WEIGHTS-SPEC.md) for the complete binary specification and rank-sweep (`r = 8 -> 32 -> 64 -> 128`) Pareto analysis.
4. **Embodied AI, HMI Biosignals & Living Metabolism (`embodied.rs`, `living.rs`, `sigql`):**
   - Entity-Component-Time (`EctStore`) with `HLC` + `LSL` clock calibration, `kdb+`-class strictly causal `ASOF` sensor alignment (`1,000` joins over `50,000` `400 Hz` ticks in `633 µs`, `0` future leakage), `ROS2 tf2`-class `SE(3)` quaternion `SLERP` `TransformTree`, `SigQL` FFT/Butterworth/Hilbert biosignal queries, active-inference `SurpriseGate`, and Feldera-class `DBSP` `ZSet<T>` $O(|\Delta|)$ incremental views.
5. **Post-Docker `JouleCapsule`, Post-Git `JouleSpace`, `JouleKube` & 6-Rail Energy Fabric (`joule-db-branch`, `joule-db-energy`, `joule-db-ledger`):**
   - Daemonless pure-Rust OCI v2 image pulls + $O(1)$ CoW container+DB branching (`1.13 µs/branch` across `1,000` sandboxes), `Jujutsu`/`Zed Delta` continuous op-logs + LiveKit-class `turn_epoch` barge-in rooms, `JouleWard` RFC-2104 `ThermoCap` macaroons (`Seatbelt`/`Landlock`/`WASI 0.2`), and **`JouleKube`** (`kubectl`-compatible Kubernetes + CNCF AI-Conformance `DRA`/`Kueue`/`LWS`/`InferencePool`/`agent-sandbox`/`Kagent` backed by MVCC SQL `_jouledb_kube_objects` at `0.0W` warm-pool standby).
   - Certified live on **Apple M5 Max Metal GPU** (`104.47 µJ` vs `541.07 µJ` CPU = **`5.18x` hardware joule savings**) and across 11 hardware/runtime targets (`macOS ARM64/x86_64`, `Linux CUDA/ROCm-RADV/ARM64`, `Windows 11`, `Android Snapdragon`, `wasm32-wasip2`, Browser `OPFS+WebGPU`, C99 FFI, and `#![no_std]` edge).

---

## 2. Unified 7-Pillar Industry Benchmark Proof (`POST /api/v1/bench/industry_suite`)

| Pillar | Industry Standard | Verified Apple M5 Max Result (`v0.2.9`) |
| :--- | :--- | :--- |
| **1. OLTP Concurrency** | `TPC-C` under Cahill `SSI` | **`106,383 TPS`**, `ytd_conserved = true` |
| **2. Columnar Analytics** | `ClickBench` / `Vortex` `ZoneMap + SelectionVector` + `Morsel` Spill | **`97.96%` cold-granule skip in `164 µs`**; `15k ⋈ 3k` Grace spill under `16 KiB` RAM |
| **3. Financial Accounting & DST** | `TigerBeetle` 2-Phase Linked Ledger + `VOPR` Simulator | **`3,200+` transfers**, `16` torn WALs + `16` NAND bit-flips caught, **`0` conservation errors** |
| **4. Wall Street / Hyperscaler DAGs** | Goldman `SecDB` Dirty-Cone + Google `Napa` Freshness SLA | **`99.23%` compute saved (`35 µs` / `43 nJ`)**; `150` burst writes coalesced to `1` refresh |
| **5. Filtered Vector Search** | `Big-ANN` `RaBitQ` 1-Bit `POPCNT` + `ACORN` 2-Hop | **`~18.3x–32x` compression vs `FP32`**, `145 µs` selective filtered recall hit |
| **6. Streaming Incremental SQL** | `Nexmark` / Feldera `DBSP` `ZSet<T>` | **`171 µs`** incremental $O(|\Delta|)$ equi-join + aggregation |
| **7. Formal Isolation & Logic Oracles** | Kleppmann `Hermitage` + Jepsen/`Elle` + `SQLancer` | **`6 / 6` `Hermitage` clean (`1.13 ms`)**, **`0` `Elle` `SCC` cycles**, **`0` `SQLancer` violations** |

---

## 3. Honest Engineering Boundaries

1. **Self-hosted open-source engine, not a managed SaaS SLA:** Multi-node Raft, 2PC, `JouleKube`, and Heterogeneous Fabric ship in code; OpenIE does not operate a commercial hosted cloud from this repository.
2. **Eckart–Young rank bounds on `.jsyn`:** Ultra-compact `r = 8..16` (`0.02–0.44 BPW`) `.jsyn` checkpoints preserve principal singular subspaces while delegating encyclopedic fact storage to MVCC SQL + `Engram` + `TriModal GraphRAG`; higher standalone open-domain generation fidelity scales continuously via `target_rank = 32..256` + `AWQ ⊕ GPTQ SVF` calibration.
3. **OS kernel sandbox primitives:** Full `cgroups v2` / `Landlock` / `seccomp-bpf` isolation requires Linux; macOS uses `Seatbelt` SBPL + unprivileged CoW VFS confinement; portable targets use `WASI 0.2` fuel/memory caps.
